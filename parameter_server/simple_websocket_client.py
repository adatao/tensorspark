#simple websocket client

import tornado.websocket
from tornado import gen 

@gen.coroutine
def test_ws():
    client = yield tornado.websocket.websocket_connect("ws://localhost:8888/")
    client.write_message("Testing from client")
    msg = yield client.read_message()
    print("msg is %s" % msg)
    yield client.read_message()
    print("msg is %s" % msg)
#    msg = yield client.read_message()
#    print("msg is %s" % msg)
#    client.close()

#if __name__ == "__main__":
#    tornado.ioloop.IOLoop.instance().run_sync(test_ws)