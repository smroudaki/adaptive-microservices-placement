class NetworkEdge:
    def __init__(self, source, target, PR, BW):
        self.source = source
        self.target = target
        self.PR = PR
        self.BW = BW
        self.name = f"{source}->{target}"
        self.id = None
