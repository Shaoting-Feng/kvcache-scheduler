import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np

from flowsim.buffer import Buffer
from flowsim.common import Document, get_cur_bw, usleep

TRACE_DIR: Path = Path(__file__).parent.parent / "trace"
LOG_DIR = Path(__file__).parent.parent / "log"

SEND2BUFFER_BW: pd.DataFrame = pd.read_csv(TRACE_DIR / "send2buf_bw.csv")
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

    def gaussian_transition(self, current_doc_id: int, sigma: float) -> int:
        """
        Calculate the next doc_id based on a Gaussian distribution.
        
        Args:
            current_doc_id (int): The current doc_id.
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            int: The next doc_id.
        """
        # Generate the next doc_id using Gaussian distribution centered on current_doc_id
        next_doc_id = int(np.random.normal(loc=current_doc_id, scale=sigma))
        return max(0, next_doc_id)  # Ensure doc_id is non-negative

     def run_with_gaussian(self, sigma: float = 1.0) -> None:
        """
        Run the sender using Gaussian-distributed doc_id transitions.

        Args:
            sigma (float): Standard deviation for the Gaussian distribution.
        """
        current_doc_id = self._trace.iloc[0]['doc_id']  # Start with the first doc_id in trace
        
        for _ in range(len(self._trace)):
            # Calculate the next doc_id
            next_doc_id = self.gaussian_transition(current_doc_id, sigma)

            # Ensure next_doc_id exists in the trace
            if next_doc_id in self._trace['doc_id'].values:
                self.send_doc(next_doc_id, version=1)  # Send the document
                current_doc_id = next_doc_id  # Update for the next iteration

            # Optional: Add a delay between sends
            usleep(int(50000))
    
    def run(self, strategy: str = "all_at_once") -> None:
        ####################### STUDENT CODE STARTS HERE ######################

        if strategy == "first_come_first_serve":
            # First-Come-First-Serve Strategy:
            # Documents are sent in the order they arrive in the trace file,
            # ensuring that the first document in the list is sent first.
            for _, row in self._trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1)
                
        elif strategy == "shortest_gen_latency_first":
            # Shortest Generation Latency First:
            # Sorts documents by their generation latency (v1_lat) and sends
            # those with the shortest latency first to maximize responsiveness.
            sorted_trace = self._trace.sort_values(by='v1_lat')
            for _, row in sorted_trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1)

        elif strategy == "shortest_job_first":
            sorted_trace = self._trace.sort_values(by='v1_size')
            for _, row in sorted_trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1)

        elif strategy == "random":
            # Random Strategy:
            # Shuffles the document list randomly, then sends each document in random order.
            shuffled_trace = self._trace.sample(frac=1).reset_index(drop=True)
            for _, row in shuffled_trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1)

        elif strategy == "highest_quality_first":
            # Highest Quality First:
            # Sends documents in descending order of quality score, prioritizing
            # those with the highest quality. Useful when quality is critical.
            sorted_trace = self._trace.sort_values(by='v1_score', ascending=False)
            for _, row in sorted_trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1)

        elif strategy == "adaptive_version":
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

        elif strategy == "round_robin_version":
            # Round Robin Version Selection:
            # Cycles through v1, v2, and v3 versions for each document in a round-robin fashion.
            # Useful for testing all versions and balancing load across different quality levels.
            versions = [1, 2, 3]
            for idx, row in enumerate(self._trace.iterrows()):
                version = versions[idx % len(versions)]
                usleep(int(50000))
                self.send_doc(row.doc_id, version)

        elif strategy == "random_version":
            # Random Version Selection:
            # Randomly selects a version (v1, v2, or v3) for each document.
            # This approach introduces randomness to simulate unpredictable network conditions.
            import random
            for _, row in self._trace.iterrows():
                version = random.choice([1, 2, 3])
                usleep(int(50000))
                self.send_doc(row.doc_id, version)

        elif strategy == "random_version_sjf":
            # Randomly selects a version (v1, v2, or v3) for each document, 
            # then sorts by the size of the selected version (Shortest Job First).
            new_trace = self._trace.copy()
            new_trace['random_choice'] = np.random.choice([1, 2, 3], size=len(new_trace))
            new_trace['selected_size'] = new_trace.apply(
                lambda row: row['v1_size'] if row['random_choice'] == 1 else
                            row['v2_size'] if row['random_choice'] == 2 else
                            row['v3_size'],
                axis=1
            )
            sorted_trace = new_trace.sort_values(by='selected_size')
            for _, row in sorted_trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, row.random_choice)

        elif strategy == "equal_bandwidth_share":
            # Equal Bandwidth Sharing Strategy:
            # Divides the available bandwidth equally among all documents,
            # ensuring each document gets an equal portion of the bandwidth.
            num_docs = len(self._trace)
            bw_share = 1 / num_docs
            for _, row in self._trace.iterrows():
                usleep(int(50000))
                self.send_doc(row.doc_id, 1, bw_share=bw_share)

        elif strategy == "weighted_fair_share":
            # Weighted Fair Share:
            # Assigns bandwidth based on document generation latency.
            # Documents with higher latency get more bandwidth to reduce
            # their time in the system, balancing the transmission load.
            total_lat = self._trace['v1_lat'].sum()
            for _, row in self._trace.iterrows():
                bw_share = row['v1_lat'] / total_lat
                usleep(int(50000))
                self.send_doc(row.doc_id, 1, bw_share=bw_share)
        elif strategy == "possion":
            shuffled_trace = self._trace.sample(frac=1).reset_index(drop=True)
            for _, row in shuffled_trace.iterrows():
                # Use Poisson distribution for timing
                poisson_interval_us = int(np.random.poisson(mean_interval_ms) * 1000)
                if poisson_interval_us > 0:
                    usleep(poisson_interval_us)
                self.send_doc(row.doc_id, 1)
        else:
            raise ValueError("Unsupported strategy provided")

        ######################## STUDENT CODE ENDS

if __name__ == "__main__":
    # Initialize Buffer and Sender with relative paths
    buffer = Buffer()
    trace_file = Path(__file__).parent.parent / "trace" / "doc_stats.csv"
    sender = Sender(buffer, trace_file)

    # Choose a strategy: "all_at_once", "equal_bandwidth_share", or "first_come_first_serve"
    strategy = "adaptive_version"  # Change this as needed

    # Run sender with the selected strategy
    sender.run(strategy=strategy)