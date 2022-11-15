import matplotlib.pyplot as plt
import retico_core
import torch
import threading
import time

from zmq_microphone_in import TensorIU

"""
This is not working.

Matplotlib is not thread safe and complaints:
- "RuntimeError: main thread is not in main loop"

So can't use it in its  own thread and if we update in `process_update` it 
"""


class MatplotlibModule(retico_core.abstract.AbstractConsumingModule):
    @staticmethod
    def name():
        return "MatplotlibModule StereoCollector"

    @staticmethod
    def description():
        return "MatplotlibModule plots audio streams"

    @staticmethod
    def input_ius():
        return [TensorIU]

    def __init__(
        self,
        buffer_time=2,
        sample_rate=16_000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.n_samples = int(sample_rate * buffer_time)
        self.downsample = 10
        self._thread_is_active = False
        self.tensor = None

        self.i = 0

    def process_update(self, update_msg):
        tensor = None
        for iu, ut in update_msg:
            if ut == retico_core.UpdateType.ADD:
                tensor = iu.tensor  # 2, 32000

        if tensor is not None:
            self.tensor = tensor

        self.i += 1
        if self.i % 50 == 0:
            self.update_plot()

    def prepare_run(self):
        self.fig, [self.ax1, self.ax2] = plt.subplots(2, 1)
        (self.l1,) = self.ax1.plot(torch.zeros(self.n_samples // self.downsample))
        (self.l2,) = self.ax2.plot(torch.zeros(self.n_samples // self.downsample))
        self.ax1.set_ylim([-1, 1])
        self.ax2.set_ylim([-1, 1])
        self.ax1.set_xticks([])
        self.ax2.set_xticks([])
        plt.show(block=False)

        self._thread_is_active = True

    def shutdown(self):
        self._thread_is_active = False
        plt.close("all")

    def update_plot(self):
        self.l1.set_ydata(self.tensor[0, :: self.downsample])
        self.l2.set_ydata(self.tensor[1, :: self.downsample])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _thread(self):
        global fig
        global l1
        global l2
        while self._thread_is_active:
            time.sleep(0.1)
            if self.tensor is not None:
                # self.update_plot()
                self.l1.set_ydata(self.tensor[0, :: self.downsample])
                self.l2.set_ydata(self.tensor[1, :: self.downsample])
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                print("thread update")


if __name__ == "__main__":
    import time

    mpl = MatplotlibModule()
    mpl.prepare_run()

    for i in range(10):
        mpl.update_plot(torch.randn((2, mpl.n_samples)) / 2)
        time.sleep(0.5)
        print(f"step {i}")
