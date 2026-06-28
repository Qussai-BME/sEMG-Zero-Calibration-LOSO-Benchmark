"""
Checkpointing utilities to resume interrupted validation.
"""

import os
import pickle

class Checkpoint:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.load() if os.path.exists(filename) else {}

    def load(self):
        with open(self.filename, 'rb') as f:
            return pickle.load(f)

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.data, f)

    def update(self, key, value):
        self.data[key] = value
        self.save()

    def get(self, key, default=None):
        return self.data.get(key, default)

    def contains(self, key):
        return key in self.data