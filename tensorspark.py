# outer configuration, called by tensorflow user
# i.e.
# tensorspark.set_hyperparameters()
# tensorspark.run_model_on_spark(my_model)
# it exits the python context only to re-enter it
# bc spark likes to be started from command line
# spark-submit tensorspark.py
# which runs the code outside of functions at the bottom here...


# this records the experiments in a database
# so we can have a record of how everything went
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment('tensorflow001')
ex.observers.append(MongoObserver.create(db_name='tensorspark_experiments'))

# @ex.config tells sacred to make all variables in this function
# rewritable from the experiment context
# it also lets us inject these variables into any function labeled
# @ex.capture
# Note: you cannot return anything in this function or it breaks with
# an unhelpful error message :(

@ex.config
def set_hyperparameters():
   memory_per_executor = '10G'
   #cluster_manager_url = 'spark://192.168.1.91:7077'
   cluster_manager_url = 'local[*]'
   total_cores = 12


from subprocess import call
# this means that unused arguments are automatically supplied by
# the experiment context
@ex.capture
def run_model_on_spark(model, memory_per_executor, cluster_manager_url, total_cores):
   args = ['spark-submit']
   args.append('--master %s' % cluster_manager_url)
   args.append('--executor-memory %s' % memory_per_executor)
   args.append('--total-executor-cores %d' % total_cores)
   args.append('tensorspark.py')
   call(args)
   # now it exits python, makes a system call that calls spark on this file
   # could be another file later




#code for driver
import pyspark
print 'running pyspark'
# this is just bad placeholder code
#sc.map(lambda x: x[my_columns] if sampled(x))
#sc.map(testing_samples)
#sc.reduce(lambda x: get_predictions_for(x))


#code for worker
import tensorflow as tf
# ?