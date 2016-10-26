#Distributed TensorFlow on Spark 
##TensorSpark productionalized in yarn-cluster mode
This fork contains modifications/improvements that are mostly relevant to someone interested in taking TensorSpark to production in yarn-cluster mode with CPU machines.

###Summary of changes
There are few minor improvements (see commits for details) and the following 2 major changes:
* tensorspark.py: Reading the testset from the HDFS instead (Avoiding the need to put the testset on local disk; we are putting training and test sets at the same location on the HDFS)
* parameterwebsocketclient.py: Find the machine that gets the Spark Driver in yarn-cluster mode (either way, there are some configs to be done here)

###To run
1. zip pyfiles.zip ./parameterwebsocketclient.py ./parameterservermodel.py ./mnistcnn.py ./mnistdnn.py ./moleculardnn.py ./higgsdnn.py
2. spark-submit \
<br />--master yarn \
<br />--deploy-mode cluster \
<br />--queue default \
<br />--num-executors 3 \
<br />--driver-memory 20g \
<br />--executor-memory 60g \
<br />--executor-cores 8 \
<br />--py-files ./pyfiles.zip \
<br />./tensorspark.py

##Original Arimo ReadMe:
A repo for Distributed TensorFlow on Spark, first presented at the 2016 Spark Summit East 

Slide deck: http://www.slideshare.net/arimoinc/distributed-tensorflow-scaling-googles-deep-learning-library-on-spark-58527889
Video of talk will be posted when available.

Project layout:

<br>tensorspark/gpu_install.sh - script to build tf from source with gpu support for aws
<br>tensorspark/simple_websocket_*.py - simple tornado websocket example
<br>tensorspark/parameterservermodel.py - "abstract" model class that has all tensorspark required methods implemented
<br>tensorspark/*dnn.py - specific fully connected models for specific datasets
<br>tensorspark/mnistcnn.py - convolutional model for mnist
<br>tensorspark/parameterwebsocketclient.py - spark worker code
<br>tensorspark/tensorspark.py - entry point and spark driver code

to run

zip pyfile.zip ./parameterwebsocketclient.py ./parameterservermodel.py ./mymodel.py
spark-submit ..... --pyfiles ./pyfiles.zip ./tensorspark.py


