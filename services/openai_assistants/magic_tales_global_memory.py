
class GlobalMemory:
    def __init__(self):
        self.data = {}

    def update(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)

    def clear(self):
        self.data.clear()

    def to_dict(self):
        return self.data.copy()