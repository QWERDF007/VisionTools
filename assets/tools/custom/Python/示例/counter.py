class CounterTool:
    def __init__(self):
        self.count = 0
    def initialize(self, params):
        self.count = 0
        return {'__status_code': 0, '__status_message': ''}
    def execute(self, inputs):
        self.count += 1
        return {'__status_code': 0, '__status_message': '', 'count': self.count}
    def release(self):
        return {'__status_code': 0, '__status_message': ''}
