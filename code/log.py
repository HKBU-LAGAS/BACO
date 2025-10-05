import os
import sys
import json
import numpy as np

class Logger(object):
    def __init__(self, path, name, mode=None):
        self.summary = {}
        log_file = os.path.join(path, name)
        print('saving log to ', log_file)
        self.terminal = sys.stdout
        self.file = None
        self.open(log_file,mode)

    def open(self, file, mode=None):
        if mode is None:
            mode = 'w'
        self.file = open(file, mode)

    def write(self, *args, sep=' ', end='\n', is_terminal=1, is_file=1):
        message = sep.join(str(a) for a in args)
        if not message.endswith(end):
            message += end
        # if '\r' in message:
        #     is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def write_summary(self, add_dict):
        self.summary.update(add_dict)

    def _convert_np(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_np(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    def print_summary(self):
        print('Run summary:')
        print(json.dumps(
            self._convert_np(self.summary),
            indent=2,
            ensure_ascii=False
        ))

    def write_args(self, args):
        message = ''
        for arg in vars(args):
            message += arg + '=' + str(getattr(args, arg)) + '\t'

    def close(self):
        self.file.close()