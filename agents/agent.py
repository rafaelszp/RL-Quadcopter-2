class QuadcopterAgent:

    def __init__(self,task):
        self.task = task

    def act(self):
        return [i/12 for i in range(1,13)]