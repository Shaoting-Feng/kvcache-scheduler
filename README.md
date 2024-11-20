This is the flow scheduling simulator for the final project for CMSC 333: Graduate Computer Networking @ UChicago. 

### Structure of codebase
- flowsim: the directory that contains the classes for the **sender**, **buffer** and **receiver** 
- log: the directory that contains the logs of running the traces 
- trace: the directory that contains the traces and some metadata

### How to install 

In order to run the code, you need to install several packages.

```
pip install pandas
```

Also, please add the path to flowsim to python path by: 
```
export PYTHONPATH=$PYTHONPATH:<PATH TO THIS REPO>/flowsim
```
**Remember to replace <PATH TO THIS REPO> to your own path to this repository!**

An example run: 

```
python run.py
```
Check the hit rate, quality score, average retrieval time in log/receiver.log. 


### Where to add your code

- Sending function @ the sender: add your implementation  in the ```run``` function. To send a KV cache of a specific ```doc_id```, and a specific compression version ```version```, call the ```send_doc(doc_id, version)``` function which will send the KV cache to buffer. Note that the ```send_doc``` is an asynchronous function. 
- Sending function @ the buffer: add your implementation in the ```_dispatch``` function. When the receiver send a request to the buffer, if the KV cache is present, then the request will be submitted to the job queue.
Your job in the ```_dispatch``` function is to select one from the job queue when there are many, to respond first. 
A potentially useful function is ```enqueue_time``` function which shows how long the job has been in the job queue. 
