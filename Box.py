class Box():
    def __init__(self, position=[0, 0, 0, 0], rank=0):
        self.position = position
        self.rank = rank
    
    def update(self, position, rank):
        self.position = position
        self.rank = rank