import pathlib
import json
import time
from threading import Thread

from flowsim import Buffer, Receiver, Sender

TRACE_DIR = pathlib.Path(__file__).parent / "trace"

# Use timestamp to avoid overwriting the result file
timestamp = time.strftime("%Y%m%d_%H%M%S")
file_name = f"log/result_{timestamp}.json"

trace = "poisson-600ms.trace"
sender_strategy = "first_come_first_serve"
buffer_startegy = "fifo"

# Record the configuration
with open(file_name, mode='w') as file:
    data = {'trace': trace, 'sender_strategy': sender_strategy, 'buffer_scheduler_value': buffer_startegy, 'miss_rate': None, 'quality_score': None, 'retrieval_time': None} 
    file.seek(0)
    json.dump(data, file, indent=4)

def main():
    buf = Buffer(file_name, buffer_startegy)
    receiver = Receiver(buf, TRACE_DIR / trace, file_name)
    sender = Sender(buf, TRACE_DIR / "doc_stats.csv")

    receiver_thread: Thread = Thread(target=receiver.trace_replay)
    sender_thread: Thread = Thread(target=sender.run, kwargs={"strategy": sender_strategy})
    sender_thread.daemon = True

    sender_thread.start()
    receiver_thread.start()

    receiver_thread.join()


if __name__ == "__main__":
    main()
