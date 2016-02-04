#simple websocket

import tornado.websocket
import tornado.ioloop
import tornado.web

class WSHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        print 'new connection'

    def on_message(self, message):
        print 'message received %s' % message
        self.write_message(message)

    def on_close(self):
    	print 'connection closed'


application = tornado.web.Application([
  (r'/', WSHandler),
])


if __name__ == "__main__":
    app = tornado.web.Application([(r"/", WSHandler)])
    app.listen(8888)
    tornado.ioloop.IOLoop.instance().start()