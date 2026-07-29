from .connector import MilvusConnector


class MilvusContext:
    def __init__(self, connector: MilvusConnector):
        self.connector = connector
