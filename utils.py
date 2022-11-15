import retico_core


class TensorIU(retico_core.abstract.IncrementalUnit):
    @staticmethod
    def type():
        return "Tensor IU"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.channels = 2
        self.tensor = None

    def set_tensor(self, tensor):
        self.tensor = tensor
