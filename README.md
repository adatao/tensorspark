A repo for the Spark-Summit East project, TensorFlow on Spark

Project layout:

tensorspark/gpu_install.sh - script to build tf from source with gpu support for aws
tensorspark/simple_websocket_*.py - simple tornado websocket example
tensorspark/parameterservermodel.py - "abstract" model class that has all tensorspark required methods implemented
tensorspark/*dnn.py - specific fully connected models for specific datasets
tensorspark/mnistcnn.py - convolutional model for mnist
tensorspark/parameterwebsocketclient.py - spark worker code
tensorspark/tensorspark.py - entry point and spark driver code

to run

zip pyfile.zip ./parameterwebsocketclient.py ./parameterservermodel.py ./mymodel.py
spark-submit ..... --pyfiles ./pyfiles.zip ./tensorspark.py


