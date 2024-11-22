import pathlib
import json
import time
from threading import Thread

from flowsim import Buffer, Receiver, Sender, SlidingWindow

TRACE_DIR = pathlib.Path(__file__).parent / "trace"

# Use timestamp to avoid overwriting the result file
timestamp = time.strftime("%Y%m%d_%H%M%S")
file_name = f"log/result_{timestamp}.json"

# The parameters that need to be changed
trace = "poisson-5000ms.trace"
sender_strategy = "sliding_sliding"
buffer_startegy = "fifo"
sliding_window_size = 10

# Record the configuration
with open(file_name, mode='w') as file:
    data = {'trace': trace, 'sender_strategy': sender_strategy, 'buffer_scheduler_value': buffer_startegy, 'miss_rate': None, 'quality_score': None, 'retrieval_time': None} 
    file.seek(0)
    json.dump(data, file, indent=4)

def main():
    sliding_window = SlidingWindow(sliding_window_size)

    buf = Buffer(file_name, buffer_startegy, sliding_window)
    receiver = Receiver(buf, TRACE_DIR / trace, file_name)
    sender = Sender(buf, TRACE_DIR / "doc_stats.csv", sliding_window)

    receiver_thread: Thread = Thread(target=receiver.trace_replay)
    sender_thread: Thread = Thread(target=sender.run, kwargs={"strategy": sender_strategy})
    sender_thread.daemon = True

    sender_thread.start()
    receiver_thread.start()

    receiver_thread.join()


if __name__ == "__main__":
    main()
