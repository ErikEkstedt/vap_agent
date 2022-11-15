import retico_core
import torch
import threading
import time

from utils import TensorIU

# model imports
from vap.model import VAPModel
from vap.utils import everything_deterministic


everything_deterministic()
torch.manual_seed(0)


class VapModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "VapModule"

    @staticmethod
    def description():
        return "A module that collects audio from two microphone inputs."

    @staticmethod
    def input_ius():
        return [TensorIU]

    def __init__(
        self,
        checkpoint="",
        buffer_time=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.sample_rate = 16_000
        self.buffer_time = buffer_time
        self.n_samples = int(buffer_time * self.sample_rate)

        self.load_model(
            "../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt"
        )
        self.x = torch.zeros((2, self.n_samples), device=self.device)

    def load_model(self, checkpoint):
        print("Load Model...")
        self.model = VAPModel.load_from_checkpoint(checkpoint)
        self.model = self.model.eval()
        self.sample_rate = self.model.sample_rate
        self.frame_hz = self.model.frame_hz
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
            print("CUDA")

    def reset(self):
        self.x = torch.zeros((2, self.n_samples))

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut != retico_core.UpdateType.ADD:
                continue
            self.x = iu.tensor

    def _thread(self):
        while self._thread_is_active:
            time.sleep(0.4)
            out = self.model.output(self.x.unsqueeze(0).to(self.device))
            pp = out["p"][0, -10:, 1].mean().item()
            p = int(50 * pp)
            print("Speaker B -> ", "#" * p)

    def prepare_run(self):
        self._thread_is_active = True
        threading.Thread(target=self._thread).start()

    def shutdown(self):
        self._thread_is_active = False
        self.reset()
