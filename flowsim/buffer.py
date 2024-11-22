import logging
from pathlib import Path
from threading import Condition, Event, RLock, Thread
from typing import Dict, List, Optional
from time import monotonic_ns
import json
import pandas as pd

from flowsim.common import Document, get_cur_bw, usleep
from flowsim.slidingwindow import SlidingWindow

LOG_DIR: Path = Path(__file__).parent.parent / "log"
TRACE_DIR: Path = Path(__file__).parent.parent / "trace"

BUF2RECV_BW: pd.DataFrame = pd.read_csv(TRACE_DIR / "buf2recv_bw.csv")
ST = monotonic_ns()

class Request:
    def __init__(self, id: int) -> None:
        self.id = id
        self.resp: Optional[Document] = None
        self.submit_time: int = monotonic_ns() - ST
        self._done: Event = Event()
        self.bandwidth_share = 1

    def wait_for_resp(self) -> Optional[Document]:
        self._done.wait()
        # fake transmission delay from buffer to receiver
        if self.resp is not None:
            usleep(int(self.resp.size * 8 / get_cur_bw(BUF2RECV_BW) * 1e6 * self.bandwidth_share))
        return self.resp

    def respond(self, bs) -> None:
        self.bandwidth_share = bs
        self._done.set()

    def enqueue_time(self) -> int:
        return monotonic_ns() - ST - self.submit_time

    def write_doc(self, resp: Optional[Document]) -> None:
        self.resp = resp


class Buffer:
    def __init__(self, file_name: str, strategy: str, sliding_window: SlidingWindow) -> None:
        self.buffer_scheduler_value = strategy
        self._cache_store: Dict[int, Document] = {}
        self._buf_lock: RLock = RLock()
        self._job_queue: List[Request] = []
        self._is_empty: Condition = Condition()
        self._worker_thread = Thread(target=self._worker)
        self._worker_thread.daemon = True
        self._worker_thread.start()

        self._logger = logging.getLogger(self.__class__.__name__)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(LOG_DIR / "buffer.log", mode="w")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.INFO)
        self.sliding_window = sliding_window

    def _wait_not_empty(self) -> None:
        with self._is_empty:
            self._is_empty.wait_for(lambda: len(self._job_queue) > 0)

    def _submit_for_sched(self, req: Request) -> None:
        self._job_queue.append(req)
        with self._is_empty:
            self._is_empty.notify()

    def _dispatch(self, scheduler: str) -> List[Request]:
        self._wait_not_empty()

        if scheduler == "fifo":
            return self.fifo_scheduling()
        elif scheduler == "concurrent":
            return self.share_bandwidth()
        elif scheduler == "sjf":
            return self.sjf_scheduling()

    def _worker(self):
        while True:
            req_list: List[Request] = self._dispatch(self.buffer_scheduler_value)  # get the requests to be handled

            total_size = sum(req.resp.size for req in req_list)
            
            # Store threads for each request
            threads = []
            for req in req_list:
                # Create a thread for each request
                thread = Thread(target=req.respond, args=(req.resp.size / total_size,))
                threads.append(thread)
                thread.start()  

            # 　Wait for all threads to complete
            for thread in threads:
                thread.join()

    def put(self, doc: Document) -> None:
        """Add a document KV Cache to the buffer."""
        self._logger.info(
            f"Buffer received document {doc.id:.0f}"
            f" with quality score {doc.quality}."
        )
        self._cache_store[doc.id] = doc

    def get(self, id: int) -> Optional[Document]:
        """Get a document from the buffer.

        This function blocks until the query is dispatched by the scheduler.

        Args:
        id (int): Document id to be retrieved.
        """
        # directly return miss if sender has not generate the KV cache for
        # requested doc by the time of request arrival
        if id not in self._cache_store:
            self.sliding_window.add(id, 0)
            return None

        req: Request = Request(id)

        with self._buf_lock:
            doc: Optional[Document] = self._cache_store.get(req.id)
            req.write_doc(doc)

        self._submit_for_sched(req)
        resp: Optional[Document] = req.wait_for_resp()
        self.sliding_window.add(id, resp.quality if resp is not None else 0)
        return resp

    def fifo_scheduling(self) -> Request:
        request_list = []
        request = self._job_queue.pop(0)
        self._logger.info(
            f"Request enqueue time {request.enqueue_time() / 1e6} ms"
        )
        request_list.append(request)
        return request_list

    def sjf_scheduling(self) -> Request:
        request_list = []
        shortest_request = min(self._job_queue, key=lambda req: req.resp.size)
        self._job_queue.remove(shortest_request)
        self._logger.info(
            f"Request enqueue time {shortest_request.enqueue_time() / 1e6} ms"
        )
        request_list.append(shortest_request)
        return request_list

    def share_bandwidth(self) -> Request:
        request_list = []
        for request in self._job_queue:
            self._job_queue.remove(request)
            self._logger.info(
                f"Request enqueue time {request.enqueue_time() / 1e6} ms"
            )
            request_list.append(request)
        return request_list