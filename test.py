import os
import json
from collections import defaultdict

class Logger(object):
    def __init__(self):
        super(Logger, self).__init__()
        self.logs = defaultdict(list)
        self.logs['epoch'] = 0

    def log(self, key, value):
        if isinstance(value, dict):
            for k, v in value.items():
                self.log('{}.{}'.format(key, k), v)
        else:
            self.logs[key].append(value)
    
    def pr(self):
        print(self.logs)

log = Logger()
log.logs['test_loss'].append(0)
log.logs['test_loss']

new_dict = {'1':[3,4], 0:5}
log.log('new', new_dict)
log.logs
