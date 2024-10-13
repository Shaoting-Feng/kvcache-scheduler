import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from flowsim.buffer import Buffer
from flowsim.common import Document, get_cur_bw, usleep

TRACE_DIR: Path = Path(__file__).parent.parent / "trace"
LOG_DIR = Path(__file__).parent.parent / "log"

BUF2RECV_BW: pd.DataFrame = pd.read_csv(TRACE_DIR / "send2buf_bw.csv")
N_CHUNKS = 100
MAX_CONCURRENT_SEND = 4


class _AtomicInteger:
    def __init__(self, value=0):
        self._value = int(value)
        self._lock = threading.Lock()

    def inc(self, d=1):
        with self._lock:
            self._value += int(d)
            return self._value

    def dec(self, d=1):
        return self.inc(-d)

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, v):
        with self._lock:
            self._value = int(v)
            return self._value


class Sender:
    def __init__(self, buf: Buffer, trace_file: str | Path) -> None:
        self._buf: Buffer = buf
        self._trace: pd.DataFrame = pd.read_csv(trace_file)
        self._n_ongoing_send: _AtomicInteger = _AtomicInteger(0)
        self._workers = ThreadPoolExecutor(MAX_CONCURRENT_SEND)

        self._logger = logging.getLogger(self.__class__.__name__)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(LOG_DIR / "sender.log", mode="w")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.INFO)

    def _submit_sending(self, doc: Document) -> None:
        real_gen_lat: float = 0
        real_trans_lat: float = 0
        self._n_ongoing_send.inc()
        sz_per_chunk = doc.size / N_CHUNKS
        lat_per_chunk = doc.gen_lat / N_CHUNKS
        for _ in range(N_CHUNKS):
            chunk_gen_lat = lat_per_chunk * 1e3 * self._n_ongoing_send.value
            chunk_trans_lat = (
                sz_per_chunk
                / get_cur_bw(BUF2RECV_BW)
                * 8
                * self._n_ongoing_send.value
                * 1e6
            )
            real_gen_lat += chunk_gen_lat
            real_trans_lat += chunk_trans_lat
            usleep(int(chunk_gen_lat + chunk_trans_lat))
        self._buf.put(doc)
        self._n_ongoing_send.dec()

        self._logger.info(
            f"Sent doc {int(doc.id):.0f} with generation latency"
            f" {real_gen_lat / 1e3:.2f} ms and transmission"
            f" latency {real_trans_lat / 1e3:.2f} ms."
        )

    def send_doc(self, doc_id: int, version: int) -> Future:
        """Send a document to the buffer.

        The function submits a document to the buffer. Note that the function
        is async and sending multiple documents at same time in background will
        slow down each document's generation and transmission since they need
        to share the GPU and transmission link.

        Args:
        doc (Document): Document to be sent.
        """
        row = self._trace.loc[self._trace["doc_id"] == doc_id].iloc[0]
        if version == 1:
            doc = Document(
                id=row.doc_id,
                size=row.v1_size,
                quality=row.v1_score,
                gen_lat=row.v1_lat,
            )
        elif version == 2:
            doc = Document(
                id=row.doc_id,
                size=row.v2_size,
                quality=row.v2_score,
                gen_lat=row.v2_lat,
            )
        elif version == 3:
            doc = Document(
                id=row.doc_id,
                size=row.v3_size,
                quality=row.v3_score,
                gen_lat=row.v3_lat,
            )
        else:
            raise ValueError(f"Unsupported version id {version}")
        return self._workers.submit(self._submit_sending, doc)

    def run(self) -> None:
        ####################### STUDENT CODE STARTS HERE ######################

        # example generation scheduling: sending all doc at once and let all
        # jobs shares the GPU and transmission link.
        # replace this with your own scheduling logic
        for _, row in self._trace.iterrows():
            usleep(int(50000))
            self.send_doc(row.doc_id, 1)

        ######################## STUDENT CODE ENDS HERE #######################
