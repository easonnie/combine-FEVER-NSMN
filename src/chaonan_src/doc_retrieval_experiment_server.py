#!/usr/bin/python3.6
"""Subscribe to ros image topic and serve on the websocket
Author: chaonan99
Date: 2018/05/04
"""

from __future__ import print_function, division

import json

import tornado.httperver
import tornado.ioloop
import tornado.web
import tornado.websocket

from chaonan_src._doc_retrieval.item_rules import ItemRuleBuilderRawID
from chaonan_src._doc_retrieval.item_rules_spiral import ItemRuleBuilderSpiral
from chaonan_src._utils.wiki_pageview_utils import WikiPageviews


class FirstRun(tornado.websocket.WebSocketHandler):
    """docstring for TalkerSender"""

    def on_message(self, message):
        item = json.loads(message)
        self.application.item_rb.first_only_rules(item)
        self.write_message(item)

    def open(self):
        print("New connection for first round doc retrieval")

    def on_close(self):
        print("Connection closed for first round doc retrieval")


class FeedRun(tornado.websocket.WebSocketHandler):

    def on_message(self, message):
        self.application.item_rb.feed_sent_score_result(message)
        self.write_message("Feed sentence retrieval result successfully!")

    def open(self):
        print("New connection for feed round")

    def on_close(self):
        print("Connection closed for feed round")


class SecondRun(tornado.websocket.WebSocketHandler):

    def on_message(self, message):
        item = json.loads(message)
        item_rb = self.application.item_rb
        item_rb.second_only_rules(item)
        self.write_message(item)

    def open(self):
        print("New connection for second round doc retrieval")

    def on_close(self):
        print("Connection closed for second round doc retrieval")


def main():
    application = tornado.web.Application([
        (r'/first', FirstRun),
        (r'/feed', FeedRun),
        (r'/second', SecondRun),
    ])
    print("Reload wiki pageview ...")
    wiki_pv = WikiPageviews()
    print("Initializing ...")
    item_rb = ItemRuleBuilderSpiral()
    item_rb.wiki_pv = wiki_pv
    application.item_rb = item_rb
    print("Application started! Ready to receive doc retrieval requests!")

    http_server = tornado.httperver.HTTPServer(application)
    http_server.listen(9199)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()
