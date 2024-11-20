import logging
import pathlib
from concurrent import futures
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from time import monotonic_ns
from typing import List, Optional
import json
import pandas as pd

from flowsim.buffer import Buffer
from flowsim.common import Document, usleep

LOG_DIR = pathlib.Path(__file__).parent.parent / "log"


class Receiver:
    def __init__(self, buf: Buffer, trace_file: str | Path, file_name: str):
        self._buf: Buffer = buf
        self._trace: pd.DataFrame = pd.read_csv(
            trace_file, dtype={"ts": int, "doc_id": int}
        )
        self._qual_score: List[int] = []
        self._miss_cnt: int = 0
        self._cnt: int = 0
        self._jct: List[float] = []
        self._lock: Lock = Lock()

        self._replayer: ThreadPoolExecutor = ThreadPoolExecutor(96)
        self._query_fut: List[Future] = []
        self._all_submitted: Event = Event()

        self._logger = logging.getLogger(self.__class__.__name__)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(LOG_DIR / "receiver.log", mode="w")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.INFO)
        self.file_name = file_name

    def _submit_query(self, doc_id: int) -> None:
        self._logger.info(f"Sending query for doc {doc_id}...")
        st = monotonic_ns()
        resp: Optional[Document] = self._buf.get(doc_id)
        end = monotonic_ns()
        with self._lock:
            self._miss_cnt += 1 if resp is None else 0
            self._cnt += 1
            if resp is not None:
                self._qual_score.append(resp.quality)
                self._jct.append((end - st) / 1e6)
        if resp is None:
            self._logger.info(f"Buffer reports doc {doc_id} is not ready yet.")
        else:
            self._logger.info(
                f"Buffer returns doc {doc_id} KV Cacahe"
                f" with quality score {resp.quality}."
            )

    def trace_replay(self):
        """Replay the trace and submit queries to buffer."""
        replay_st_ns: int = monotonic_ns()

        for _, row in self._trace.iterrows():
            # trace ts is in ms, so convert it to us first
            # sleep_time_us: int = int(
            #     row.ts * 1000 - int((monotonic_ns() - replay_st_ns) / 1000)  # type: ignore
            # )
            # if sleep_time_us > 0:
            #     usleep(sleep_time_us)

            # self._query_fut.append(
            #     self._replayer.submit(self._submit_query, row.doc_id)
            # )
            poisson_interval_us = int(np.random.poisson(mean_interval_ms) * 1000)

            # Sleep for the generated interval
            if poisson_interval_us > 0:
                usleep(poisson_interval_us)

            self._query_fut.append(
                self._replayer.submit(self._submit_query, row.doc_id)
            )
        self._logger.info("All queries submitted. Waiting for all finished...")

        futures.wait(self._query_fut, return_when=futures.ALL_COMPLETED)
        self._logger.info("Benchmark completed.")
        miss_rate = self._miss_cnt / self._cnt
        self._logger.info(
            f"Miss rate: {miss_rate}"
        )
        quality_score = sum(self._qual_score) / len(self._qual_score)
        self._logger.info(
            f"Quality Score: {quality_score}"
        )
        retrieval_time = sum(self._jct) / len(self._jct)
        self._logger.info(
            "Average retrival time for cache-hit query:"
            f" {retrieval_time} ms"
        )

        # Record the result separately
        with open(self.file_name, mode='r+') as file:
            file_data = json.load(file)
            file_data['miss_rate'] = miss_rate
            file_data['quality_score'] = quality_score
            file_data['retrieval_time'] = retrieval_time
            file.seek(0)
            json.dump(file_data, file, indent=4)
