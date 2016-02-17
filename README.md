A repo for the Spark-Summit East project, TensorFlow on Spark

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


