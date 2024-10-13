import logging
from pathlib import Path
from threading import Condition, Event, RLock, Thread
from typing import Dict, List, Optional

import pandas as pd

from flowsim.common import Document, get_cur_bw, usleep

LOG_DIR: Path = Path(__file__).parent.parent / "log"
TRACE_DIR: Path = Path(__file__).parent.parent / "trace"

BUF2RECV_BW: pd.DataFrame = pd.read_csv(TRACE_DIR / "buf2recv_bw.csv")


class Request:
    def __init__(self, id: int) -> None:
        self.id = id
        self.resp: Optional[Document] = None
        self._done: Event = Event()

    def wait_for_resp(self) -> Optional[Document]:
        self._done.wait()
        # fake transmission delay from buffer to receiver
        if self.resp is not None:
            usleep(int(self.resp.size * 8 / get_cur_bw(BUF2RECV_BW) * 1e6))
        return self.resp

    def respond(self, resp: Optional[Document]) -> None:
        self.resp = resp
        self._done.set()


class Buffer:
    def __init__(self) -> None:
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

    def _wait_not_empty(self) -> None:
        with self._is_empty:
            self._is_empty.wait_for(lambda: len(self._job_queue) > 0)

    def _submit_for_sched(self, req: Request) -> None:
        self._job_queue.append(req)
        with self._is_empty:
            self._is_empty.notify()

    def _dispatch(self) -> Request:
        self._wait_not_empty()

        ####################### STUDENT CODE STARTS HERE ######################

        # example FIFO scheduling
        request = self._job_queue.pop(0)

        ######################## STUDENT CODE ENDS HERE #######################

        return request

    def _worker(self):
        while True:
            req: Request = self._dispatch()  # get the request to be handled
            with self._buf_lock:
                doc: Optional[Document] = self._cache_store.get(req.id)
                req.respond(doc)

    def put(self, doc: Document) -> None:
        """Add a document KV Cache to the buffer."""
        self._logger.info(
            f"Buffer received document {doc.id}"
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
            return None

        req: Request = Request(id)
        self._submit_for_sched(req)
        resp: Optional[Document] = req.wait_for_resp()
        return resp
