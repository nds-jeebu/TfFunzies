
class AtariBuffer:
    def __init__(self, size):
        # create the memory buffer
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, elt):
        self.data[self.end] = elt
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
