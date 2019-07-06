# -*- coding: UTF-8 -*-
"""
encapsulate postprocess logic
"""


class Postprocessor(object):
    def __init__(self, tokens):
        self.tokens = tokens

    def post_process(self, entities):
        return entities
