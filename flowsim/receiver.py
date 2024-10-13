import logging
import pathlib
from concurrent import futures
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from time import monotonic_ns
from typing import List, Optional

import pandas as pd

from flowsim.buffer import Buffer
from flowsim.common import Document, usleep

LOG_DIR = pathlib.Path(__file__).parent.parent / "log"


class Receiver:
    def __init__(self, buf: Buffer, trace_file: str | Path):
        self._buf: Buffer = buf
        self._trace: pd.DataFrame = pd.read_csv(
            trace_file, dtype={"ts": int, "doc_id": int}
        )
        self._qual_score: List[int] = []
        self._miss_cnt: int = 0
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

    def _submit_query(self, doc_id: int) -> None:
        self._logger.info(f"Sending query for doc {doc_id}...")
        st = monotonic_ns()
        resp: Optional[Document] = self._buf.get(doc_id)
        end = monotonic_ns()
        with self._lock:
            self._miss_cnt += 1 if resp is None else 0
            self._qual_score.append(resp.quality if resp is not None else 0)
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
            sleep_time_us: int = int(
                row.ts * 1000 - int((monotonic_ns() - replay_st_ns) / 1000)  # type: ignore
            )
            if sleep_time_us > 0:
                usleep(sleep_time_us)

            self._query_fut.append(
                self._replayer.submit(self._submit_query, row.doc_id)
            )
        self._logger.info("All queries submitted. Waiting for all finished...")

        futures.wait(self._query_fut, return_when=futures.ALL_COMPLETED)
        self._logger.info("Benchmark completed.")
        self._logger.info(
            f"Miss rate: {self._miss_cnt / len(self._qual_score)}"
        )
        self._logger.info(
            f"Quality Score: {sum(self._qual_score) / len(self._qual_score)}"
        )
        self._logger.info(
            "Average retrival time for cache-hit query:"
            f" {sum(self._jct) / len(self._jct)} ms"
        )
