import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
import time

from flowsim.buffer import Buffer
from flowsim.common import Document, get_cur_bw, usleep
from flowsim.slidingwindow import SlidingWindow, SlidingWindowTimeBased
from new_co import convex_optimize

TRACE_DIR: Path = Path(__file__).parent.parent / "trace"
LOG_DIR = Path(__file__).parent.parent / "log"

SEND2BUFFER_BW: pd.DataFrame = pd.read_csv(TRACE_DIR / "send2buf_bw.csv")
N_CHUNKS = 100
MAX_CONCURRENT_SEND = 4
WINDOW_UPDATE_INTERVAL = 2 * 1e6  # 10s in us

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
    def __init__(self, buf: Buffer, trace_file: str | Path, sliding_window: SlidingWindow | SlidingWindowTimeBased, sender_strategy: str, clock: list[int]) -> None:
        self._buf: Buffer = buf
        self._trace: pd.DataFrame = pd.read_csv(trace_file)
        self._n_ongoing_send: _AtomicInteger = _AtomicInteger(0)
        self._workers = ThreadPoolExecutor(MAX_CONCURRENT_SEND)
        self.sender_strategy = sender_strategy

        # Configure logging
        self._logger = logging.getLogger(self.__class__.__name__)
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(LOG_DIR / "sender.log", mode="w")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.INFO)

        # Configure separate logging for start of sending documents
        self._start_logger = logging.getLogger("SenderStartLogger")
        start_file_handler = logging.FileHandler(LOG_DIR / "sender_start.log", mode="w")
        start_file_handler.setFormatter(formatter)
        self._start_logger.addHandler(start_file_handler)
        self._start_logger.setLevel(logging.INFO)

        self.sliding_window = sliding_window
        self.clock_count = 1 # how many times the window has been updated
        self.is_next_window = True
        self.clock = clock

    def _submit_sending(self, doc: Document, version: int, bw_share: float = 1.0) -> None:
        self._start_logger.info(
            f"Starting to send doc {int(doc.id):.0f} with version v{version} "
            f"(Size: {doc.size}, Quality: {doc.quality}, Generation Latency: {doc.gen_lat} ms)"
        )
        real_gen_lat: float = 0
        real_trans_lat: float = 0
        self._n_ongoing_send.inc()
        sz_per_chunk = doc.size / N_CHUNKS
        lat_per_chunk = doc.gen_lat / N_CHUNKS
        for _ in range(N_CHUNKS):
            current_bw = get_cur_bw(SEND2BUFFER_BW) * bw_share
            chunk_gen_lat = lat_per_chunk * 1e3 * self._n_ongoing_send.value
            chunk_trans_lat = (
                sz_per_chunk / current_bw * 8 * self._n_ongoing_send.value * 1e6
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

    def send_doc(self, doc_id: int, version: int, bw_share: float = 1.0) -> Future:
        """Asynchronously send a document to the buffer."""
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
        return self._workers.submit(self._submit_sending, doc, version, bw_share)

    def send_compressed_doc(self, doc_id: int, ratio: float, bw_share: float = 1.0) -> Future:
        """Asynchronously send a document to the buffer."""
        row = self._trace.loc[self._trace["doc_id"] == doc_id].iloc[0]
        doc = Document(
            id=row.doc_id,
            size=row.v3_size * ratio,
            quality=-1.3396 * ratio * ratio + 2.3226 * ratio + 0.0038,
            gen_lat=row.v3_lat, # I found all the generation latency is the same
        )
        return self._workers.submit(self._submit_sending, doc, 0, bw_share)
    
    def run(self) -> None:

        if self.sender_strategy == "random_random":
            # Random Strategy: Shuffles the document list randomly, then sends each document in random order.
            # sending using random version
            p1 = 0.3
            p2 = 0.4
            p3 = 1 - p1 - p2

            shuffled_trace = self._trace.sample(frac=1).reset_index(drop=True)
            for _, row in shuffled_trace.iterrows():
                self.send_doc(row.doc_id, np.random.choice([1, 2, 3], p=[p1, p2, p3]))
        
        elif self.sender_strategy == "sjf_random":
            # 1. Sort all documents by the original size (Shortest Job First)
            # 2. Send documents in the sorted order with a random version
            p1 = 0.95
            p2 = 0.05
            p3 = 0

            sorted_trace = self._trace.sort_values(by='v3_size')
            for _, row in sorted_trace.iterrows():
                # usleep(int(50000)) # Uncomment this line to simulate prefill delay
                self.send_doc(row.doc_id, np.random.choice([1, 2, 3], p=[p1, p2, p3]))
        
        elif self.sender_strategy == "improvedsjf_random":
            # 1. Randomly selects a version (v1, v2, or v3) for each document
            # 2. Sort all documents by the size of the selected version (Shortest Job First)
            # 3. Send documents in the sorted order with the pre-selected version
            p1 = 0.3
            p2 = 0.4
            p3 = 1 - p1 - p2

            self._trace['random_choice'] = np.random.choice([1, 2, 3], size=len(self._trace), p=[p1, p2, p3])
            self._trace['selected_size'] = self._trace.apply(
                lambda row: row['v1_size'] if row['random_choice'] == 1 else
                            row['v2_size'] if row['random_choice'] == 2 else
                            row['v3_size'],
                axis=1
            )
            sorted_trace = self._trace.sort_values(by='selected_size')
            for _, row in sorted_trace.iterrows():
                self.send_doc(row.doc_id, row.random_choice)

        elif self.sender_strategy == "sliding_random":
            # 1. Use a sliding window to calculate the probability of which document would be requested next 
            # 2. Sort all documents by sent status, probability, and the original size
            # 3. If no document is sending, send documents in the sorted order with a random version
            std_dev = 10
            p1 = 0.3
            p2 = 0.4
            p3 = 1 - p1 - p2

            self._trace['sent'] = False
            self._trace['prob'] = 0
            while (True):
                if self._n_ongoing_send.value == 0:
                    window = self.sliding_window.window
                    x_values = np.arange(0, len(self._trace))
                    cumulative_values = np.zeros_like(x_values, dtype=np.float64)
                    for i in range(len(window)):
                        if window[i][0] != -1:
                            cumulative_values += norm.pdf(x_values, loc=window[i][0], scale=std_dev)
                    self._trace['prob'] = cumulative_values
                    sorted_trace = self._trace.sort_values(
                        by=['sent', 'prob', 'v3_size'], 
                        ascending=[True, False, True]
                    ).reset_index(drop=True)
                    if sorted_trace.iloc[0]['sent'] == False:
                        original_index = sorted_trace.iloc[0]['index'] 
                        self._trace.at[original_index, 'sent'] = True
                        doc_id = sorted_trace.iloc[0]['doc_id']
                        self.send_doc(doc_id, np.random.choice([1, 2, 3], p=[p1, p2, p3]))
                    else:
                        break

        elif self.sender_strategy == "sliding_sliding":
            # 1. Use a sliding window to calculate the probability of which document would be requested next
            # 2. Use the sliding window to calculate the sum of the received sizes
            # 3. Sort all documents by sent status, probability, and the original size
            # 4. If no document is sending, pick this document to send
            # 5. Calculte the expected version that can be received and send with this version 
            std_dev = 20

            self._trace['sent'] = False
            self._trace['prob'] = 0
            while (True):
                if self._n_ongoing_send.value == 0:
                    window = self.sliding_window.window
                    x_values = np.arange(0, len(self._trace))
                    cumulative_values = np.zeros_like(x_values, dtype=np.float64)
                    product = 0
                    window_size = 0
                    for i in range(len(window)):
                        if window[i][0] != -1:
                            cumulative_values += norm.pdf(x_values, loc=window[i][0], scale=std_dev)
                            # TODO: avoid hard-coded quality scores
                            if window[i][1] == 0.85:
                                product += self._trace.at[window[i][0], 'v1_size']
                            elif window[i][1] == 0.97:
                                product += self._trace.at[window[i][0], 'v2_size']
                            elif window[i][1] == 1:
                                product += self._trace.at[window[i][0], 'v3_size']
                            window_size += 1
                    self._trace['prob'] = cumulative_values
                    sorted_trace = self._trace.sort_values(
                        by=['sent', 'prob', 'v3_size'], 
                        ascending=[True, False, True]
                    ).reset_index(drop=True)
                    if sorted_trace.iloc[0]['sent'] == False:
                        original_index = sorted_trace.iloc[0]['index'] 
                        self._trace.at[original_index, 'sent'] = True
                        doc_id = sorted_trace.iloc[0]['doc_id']
                        product_threshold1 = self._trace.at[original_index, 'v1_size'] 
                        product_threshold2 = self._trace.at[original_index, 'v2_size']
                        if window_size == 0:
                            self.send_doc(doc_id, 1)
                        elif product / window_size > product_threshold2:
                            self.send_doc(doc_id, 3)
                        elif product / window_size < product_threshold1:
                            self.send_doc(doc_id, 2)
                        else:
                            self.send_doc(doc_id, 1)
                    else:
                        break

        elif self.sender_strategy == "convex_optimization":
            ENLARGE_FACTOR = 2
            std_dev = 3
            S_init = 2250 / 8 * WINDOW_UPDATE_INTERVAL / 1e6

            self._trace['sent'] = False
            self._trace['prob'] = 0
            sending_list = []
            product = 0
            sending_list_idx = 0
            while (True):
                time.sleep(0.1)
                if self.is_next_window:
                    self.is_next_window = False
                    window = self.sliding_window.window
                    x_values = np.arange(0, len(self._trace))
                    cumulative_values = np.zeros_like(x_values, dtype=np.float64)
                    if len(window) == 0:
                        cumulative_values = np.ones_like(x_values, dtype=np.float64)
                    else:
                        for i in range(len(window)):
                            cumulative_values += norm.pdf(x_values, loc=window[i][0], scale=std_dev)
                    self._trace['prob'] = cumulative_values
                    if product == 0:
                        S = S_init
                    else:
                        S = product * ENLARGE_FACTOR
                    sending_list = convex_optimize(self._trace, S)
                    print(sending_list)
                    sending_list_idx = 0
                    product = 0
                    self.sliding_window.clear()
                else:
                    if self._n_ongoing_send.value == 0 and sending_list_idx < len(sending_list):
                        doc_id = sending_list[sending_list_idx][0]
                        compression_ratio = sending_list[sending_list_idx][1]
                        self._trace.at[doc_id, 'sent'] = True
                        print(f"doc {doc_id} is set to True")
                        product += self._trace.at[doc_id, 'v3_size'] * compression_ratio
                        sending_list_idx += 1
                        self.send_compressed_doc(doc_id, compression_ratio)

                    # Update sliding window if needed
                    if self.clock[0] >= WINDOW_UPDATE_INTERVAL * self.clock_count:
                        self.is_next_window = True
                        self.clock_count += 1
                        print(f"Update window at {self.clock[0]}")

        elif self.sender_strategy == "first_come_first_serve":
            # First-Come-First-Serve Strategy:
            # Documents are sent in the order they arrive in the trace file,
            # ensuring that the first document in the list is sent first.
            for _, row in self._trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1)
                
        elif self.sender_strategy == "shortest_gen_latency_first":
            # Shortest Generation Latency First:
            # Sorts documents by their generation latency (v1_lat) and sends
            # those with the shortest latency first to maximize responsiveness.
            sorted_trace = self._trace.sort_values(by='v1_lat')
            for _, row in sorted_trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1)

        elif self.sender_strategy == "highest_quality_first":
            # Highest Quality First:
            # Sends documents in descending order of quality score, prioritizing
            # those with the highest quality. Useful when quality is critical.
            sorted_trace = self._trace.sort_values(by='v1_score', ascending=False)
            for _, row in sorted_trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1)

        elif self.sender_strategy == "adaptive_version":
            # Adaptive Version Based on Bandwidth:
            # Dynamically selects the document version based on current bandwidth.
            # Uses v1 (low quality) for low bandwidth, v2 for moderate bandwidth,
            # and v3 (high quality) for high bandwidth conditions.
            for _, row in self._trace.iterrows():
                current_bw = get_cur_bw(SEND2BUFFER_BW)
                if current_bw < 1700:
                    version = 1  # Low bandwidth, choose lower quality
                elif current_bw < 2200:
                    version = 2  # Moderate bandwidth
                else:
                    version = 3  # High bandwidth, choose highest quality
                
                usleep(int(50000))
                self.send_doc(row.doc_id, version)

        elif self.sender_strategy == "round_robin_version":
            # Round Robin Version Selection:
            # Cycles through v1, v2, and v3 versions for each document in a round-robin fashion.
            # Useful for testing all versions and balancing load across different quality levels.
            versions = [1, 2, 3]
            for idx, row in enumerate(self._trace.iterrows()):
                version = versions[idx % len(versions)]
                usleep(int(50000))
                self.send_doc(row.doc_id, version)

        elif self.sender_strategy == "random_version":
            # Random Version Selection:
            # Randomly selects a version (v1, v2, or v3) for each document.
            # This approach introduces randomness to simulate unpredictable network conditions.
            import random
            for _, row in self._trace.iterrows():
                version = random.choice([1, 2, 3])
                usleep(int(50000))
                self.send_doc(row.doc_id, version)

        elif self.sender_strategy == "equal_bandwidth_share":
            # Equal Bandwidth Sharing Strategy:
            # Divides the available bandwidth equally among all documents,
            # ensuring each document gets an equal portion of the bandwidth.
            num_docs = len(self._trace)
            bw_share = 1 / num_docs
            for _, row in self._trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1, bw_share=bw_share)

        elif self.sender_strategy == "weighted_fair_share":
            # Weighted Fair Share:
            # Assigns bandwidth based on document generation latency.
            # Documents with higher latency get more bandwidth to reduce
            # their time in the system, balancing the transmission load.
            total_lat = self._trace['v1_lat'].sum()
            for _, row in self._trace.iterrows():
                bw_share = row['v1_lat'] / total_lat
                usleep(int(50000))
                self.send_doc(row.doc_id, 1, bw_share=bw_share)

        else:
            raise ValueError("Unsupported strategy provided")


if __name__ == "__main__":
    # Initialize Buffer and Sender with relative paths
    buffer = Buffer()
    trace_file = Path(__file__).parent.parent / "trace" / "doc_stats.csv"
    sender = Sender(buffer, trace_file)

    # Choose a strategy: "all_at_once", "equal_bandwidth_share", or "first_come_first_serve"
    strategy = "adaptive_version"  # Change this as needed

    # Run sender with the selected strategy
    sender.run(strategy=strategy)