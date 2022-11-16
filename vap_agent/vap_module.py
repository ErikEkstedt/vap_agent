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
        refresh_time=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.refresh_time = refresh_time
        self.sample_rate = 16_000
        self.buffer_time = buffer_time
        self.n_samples = int(buffer_time * self.sample_rate)

        self.load_model(
            "../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt"
        )
        self.x = torch.zeros((1, 2, self.n_samples), device=self.device)

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

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut != retico_core.UpdateType.ADD:
                continue
            self.x = iu.tensor

    def get_speaker_probs_print(self, pp, n=100):
        p = int(n * pp)
        pn = n - p
        text = "A |"
        text += "▉" * p
        text += "-" * pn
        text += "| B"
        return text

    def get_speaker_probs_print_bins(self, pp, last_pp):

        if last_pp < 0.5:
            lim = 0.4
        else:
            lim = 0.6

        text = "<" + "-" * 30 + " OPTIONAL " + "-" * 30 + ">"
        if pp < lim:
            text = "AAAAA"
        elif pp > lim:
            text = " " * 60 + "BBBBBB"
        return text

    def _thread(self):
        print("Starting VAP-Thread")
        print("Refresh-time: ", self.refresh_time)

        n_frames = 20

        times = []
        while self._thread_is_active:
            time.sleep(self.refresh_time)
            # t = time.time()
            out = self.model.output(self.x)
            # pp = out["p_all"][0, -n_frames:, 1].mean().item()
            pp = out["p"][0, -n_frames:, 1].mean().item()
            pabc = round(100 * out["p_bc"][0, -n_frames:, 0].mean().item(), 2)
            pbbc = round(100 * out["p_bc"][0, -n_frames:, 1].mean().item(), 2)
            # last_pp = out["p"][0, :-n_frames, 1].mean().item()
            cli_text = self.get_speaker_probs_print(pp)
            # cli_t = self.get_speaker_probs_print_bins(pp, last_pp)
            print(cli_text)
            # print(f"A bc: {pabc}%")
            # print(f"B bc: {pbbc}%")
            # times.append(time.time() - t)

        # tt = torch.tensor(times)
        # u = round(tt.mean().item(), 4)
        # s = round(tt.std().item(), 4)
        # print(f"Forward: {u} +- {s}")

    def reset(self):
        pass

    def prepare_run(self):
        self._thread_is_active = True
        threading.Thread(target=self._thread).start()

    def shutdown(self):
        self._thread_is_active = False
        self.reset()
