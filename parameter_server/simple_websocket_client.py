#simple websocket client

import tornado.websocket
from tornado import gen 
import json
import numpy



@gen.coroutine
def test_ws_1():
	client = yield tornado.websocket.websocket_connect("ws://localhost:8888/")
	for i in range (10):
		client.write_message('1')
		msg = yield client.read_message()
		print msg


@gen.coroutine
def test_ws_2():
	client = yield tornado.websocket.websocket_connect("ws://localhost:8888/")
	for i in range (10):
		client.write_message('2')
		msg = yield client.read_message()
		print msg

if __name__ == "__main__":
    tornado.ioloop.IOLoop.instance().run_sync(test_ws_1)
    tornado.ioloop.IOLoop.instance().run_sync(test_ws_2)