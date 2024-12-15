This is the flow scheduling simulator. 

### Structure of codebase
- flowsim: the directory that contains the classes for the **sender**, **buffer** and **receiver** 
- log: the directory that contains the logs of running the traces 
- trace: the directory that contains the traces and some metadata

### How to install 

In order to run the code, you need to install several packages.

```
pip install pandas scipy
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
