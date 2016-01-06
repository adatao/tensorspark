import cPickle as pickle
from multiprocessing import Process, Queue
from rwlock import RWLock
import urllib2
import urlparse


class ParameterServer(Process):

    def __init__(self, serialized_model, model_deserializer, descent_func, min_updates, max_updates):
        Process.__init__(self)
        self.state = 'serving'
        self.served = 0
        self.received = 0
        self.min_updates = min_updates
        self.max_updates = max_updates
        self.lock = RWLock()
        self.pmodel = None
        self.model = model_deserializer(serialized_model)
        self.trained_model_queue = Queue(1)
        self.descent = descent_func

    def run(self):
        from flask import Flask, request

        app = Flask(__name__)

        @app.route('/')
        def index():
            return 'DeepSpark\'s parameter server'

        @app.route('/model', methods=['GET', 'POST', 'PUT'])
        def model_flask():
            i = 0
            while (self.state != 'serving' or self.served >= self.max_updates) and (i < 1000):
                import time
                time.sleep(1)
                i += 1

            # pickle on first read
            pmodel = None
            self.lock.acquire_read()
            if not self.pmodel:
                self.lock.release()
                self.lock.acquire_write()
                if not self.pmodel:
                    self.pmodel = self.model.serialize()        # pickle.dumps(self.model, -1)
                self.served += 1
                pmodel = self.pmodel
                self.lock.release()
            else:
                self.served += 1
                pmodel = self.pmodel
                self.lock.release()
            return pmodel

        @app.route('/update', methods=['GET', 'POST', 'PUT'])
        def update_flask():
            gradient = pickle.loads(request.data)

            self.lock.acquire_write()
            if self.min_updates <= self.served:
                self.state = 'receiving'
            self.received += 1

            self.descent(self.model, gradient)

            if self.received >= self.served and self.min_updates <= self.received:
                self.received = 0
                self.served = 0
                self.state = 'serving'
                self.pmodel = None

            self.lock.release()
            return '%f:%f' % (gradient['cost'], gradient['accuracy'])

        @app.route('/shutdown', methods=['GET', 'POST', 'PUT'])
        def terminate_flask():
            self.lock.acquire_read()
            self.trained_model_queue.put(self.model.serialize(), True, 3)
            self.lock.release()
            func = request.environ.get('werkzeug.server.shutdown')
            if func is None:
                raise RuntimeError('Not running with the Werkzeug Server')
            func()
            return 'OK'

        print 'Listening to 0.0.0.0:5000...'
        app.run(host='0.0.0.0', debug=True, threaded=True, use_reloader=False)

    def get_trained_model(self):
        return self.trained_model_queue.get(True, 3)


class DownpourSGD(object):
    """
    Deep Learning on Spark
    """

    def __init__(self, serialized_model, model_deserializer, gradient_func, descent_func,
                 master='127.0.0.1:5000', min_updates=0, max_updates=4096):
        """
        :param model: provide a model that can be trained in parallel on the workers
        """
        self.model = serialized_model
        self.deserializer = model_deserializer
        self.gradient = gradient_func
        self.descent = descent_func
        self.master = master
        self.min_updates = min_updates
        self.max_updates = max_updates
        self.server = None
        self.batch_size = 100

    def __enter__(self):
        self.server = ParameterServer(self.model, self.deserializer, self.descent, self.min_updates, self.max_updates)
        self.server.start()
        return self

    def __exit__(self, type, value, traceback):
        self.server.terminate()

    def get_master(self, rdd):
        master = self.master   # will be pickled
        if master is None:
            master = rdd.ctx._conf.get('spark.master')
        if master.startswith('local['):
            master = 'localhost:5000'
        else:
            if master.startswith('spark://'):
                master = '%s:5000' % urlparse.urlparse(master).netloc.split(':')[0]
            else:
                master = '%s:5000' % master.split(':')[0]
        return master

    def test(self, forward_func, rdd):
        master = self.get_master(rdd)
        print '\n*** Master: %s\n' % master

        def mapper_test(data):
            c, a, _, _, _ = forward_func(fetch_model(master=master), data)
            return [(c, a)]

        costs, acc = rdd.mapPartitions(mapper_test).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
        costs /= float(rdd.count())
        acc /= float(rdd.count())
        print 'Finished testing, accuracy = %f, costs = %f' % (acc, costs)
        shutdown_server(master)
        return costs, acc

    def train(self, rdd, loop_count):
        master = self.get_master(rdd)
        print '\n*** Master: %s\n' % master

        gradient_func = self.gradient
        iteration_id = 0

        def mapper(data):
            s = send_gradient(gradient_func(fetch_model(master=master), data, iteration_id), master=master)
            s = s.split(':')
            return [(float(s[0]), float(s[1]))]

        training_cost = []
        batch_size = self.batch_size
        for iteration_id in range(0, loop_count):
            rdd.repartition(rdd.count() / batch_size)
            costs, acc = rdd.mapPartitions(mapper).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
            costs /= float(rdd.count())
            acc /= float(rdd.count())
            training_cost.append([iteration_id, costs, acc])
            print 'Iteration %d: data size = %d, cost = %f, accuracy = %f' % (iteration_id, rdd.count(), costs, acc)

        shutdown_server(master)
        self.model = self.server.get_trained_model()
        assert self.model is not None
        return training_cost


def fetch_model(master='localhost:5000'):
    request = urllib2.Request('http://%s/model' % master,
                              headers={'Content-Type': 'application/deepspark'})
    # return pickle.loads(urllib2.urlopen(request).read())
    return urllib2.urlopen(request).read()


def send_gradient(gradient, master='localhost:5000'):
    if not gradient:
        return 'EMPTY'
    request = urllib2.Request('http://%s/update' % master, pickle.dumps(gradient, -1),
                              headers={'Content-Type': 'application/deepspark'})
    return urllib2.urlopen(request).read()


def shutdown_server(master='localhost:5000'):
    request = urllib2.Request('http://%s/shutdown' % master,
                              headers={'Content-Type': 'application/deepspark'})
    return urllib2.urlopen(request).read()