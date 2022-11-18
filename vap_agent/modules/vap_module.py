from typing import List
import retico_core
import threading
import time
import torch
from os.path import join

from vap_agent.utils import TensorIU

# model imports
from vap.model import VAPModel
from vap.utils import everything_deterministic


everything_deterministic()
torch.manual_seed(0)


CHECKPOINT = "../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt"


def vapCallback(update_msg):
    for iu, ut in update_msg:
        if ut != retico_core.UpdateType.ADD:
            continue
        print("NOW: ", iu.p_now)
        print("FUTURE: ", iu.p_future)
        print("A BC: ", iu.p_bc_a)
        print("B BC: ", iu.p_bc_b)
        print()


class VapIU(retico_core.abstract.IncrementalUnit):
    """
    Stores the output from VapModel
    """

    @staticmethod
    def type():
        return "VapModel IU"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p_now = None
        self.p_future = None
        self.p_bc_a = None
        self.p_bc_b = None
        self.H = None
        self.v1 = None
        self.v2 = None

    def set_probs(self, p_now, p_future, p_bc_a, p_bc_b, H, v1, v2):
        self.p_now = p_now
        self.p_future = p_future
        self.p_bc_a = p_bc_a
        self.p_bc_b = p_bc_b
        self.H = H
        self.v1 = v1
        self.v2 = v2


class VapModule(retico_core.AbstractModule):
    LOG_HEAD = "TIME P-NOW P-FUTURE P-BC-A P-BC-B ENTROPY V1 V2"

    @staticmethod
    def name():
        return "VapModule"

    @staticmethod
    def description():
        return "A module that collects audio from two microphone inputs."

    @staticmethod
    def input_ius():
        return [TensorIU]

    @staticmethod
    def output_iu():
        return VapIU

    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        buffer_time: float = 10,
        refresh_time: float = -0.5,
        now_lims: List[int] = [0, 1],
        future_lims: List[int] = [2, 3],
        n_frames: int = 20,
        root: str = "",
        record: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.refresh_time = refresh_time
        self.sample_rate = 16_000
        self.buffer_time = buffer_time
        self.n_samples = int(buffer_time * self.sample_rate)

        # paths logs
        self.record = record
        self.root = root
        self.filepath = join(root, "turn_taking_log.txt")

        # VapModel
        self.now_lims = now_lims
        self.future_lims = future_lims
        self.n_frames = (
            n_frames  # number of frames to consider for probabilities (the last frames)
        )

        self.load_model(self.checkpoint)
        self.x = torch.zeros((1, 2, self.n_samples), device=self.device)

    def load_model(self, checkpoint):
        print()
        print("-" * 50)
        print("Load Vap Model (pytorch)...")
        self.model = VAPModel.load_from_checkpoint(checkpoint)
        self.model = self.model.eval()
        self.sample_rate = self.model.sample_rate
        self.frame_hz = self.model.frame_hz
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.device = "cuda"
        print("Device: ", self.device)
        print("-" * 50)
        print()

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut != retico_core.UpdateType.ADD:
                continue
            self.x = iu.tensor

    def create_update_message(self, out):
        output_iu = self.create_iu()

        # v1 = out["vad"][0, -self.n_frames :, 0].mean().item()
        # v2 = out["vad"][0, -self.n_frames :, 1].mean().item()
        v1 = out["vad"][0, -5:, 0].max().item()
        v2 = out["vad"][0, -5:, 1].max().item()

        output_iu.set_probs(
            p_now=round(out["p_now"][0, -self.n_frames :, 0].mean().item(), 3),
            p_future=round(out["p_future"][0, -self.n_frames :, 0].mean().item(), 3),
            p_bc_a=round(out["p_bc"][0, -self.n_frames :, 0].mean().item(), 3),
            p_bc_b=round(out["p_bc"][0, -self.n_frames :, 1].mean().item(), 3),
            H=round(out["H"][0, -self.n_frames :].mean().item(), 3),
            v1=round(v1, 3),
            v2=round(v2, 3),
        )

        if self.record:
            self.log(output_iu)
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

    def _thread(self):
        print("Starting VAP-Thread")
        while self._thread_is_active:
            if self.refresh_time > 0:
                time.sleep(self.refresh_time)
            out = self.model.probs(self.x, now_lims=[0, 1], future_lims=[2, 3])
            update_msg = self.create_update_message(out)
            self.append(update_msg)

    def setup(self):
        self.t0 = time.time()
        if self.record:
            # add first row of txt-file
            self.logfile = open(self.filepath, "w")
            self.logfile.write(self.LOG_HEAD + "\n")

    def log(self, iu):
        """
        log IU information
        """

        if self.logfile is not None:
            t = round(self.get_current_time(), 3)
            s = f"{t}"
            s += f" {iu.p_now}"
            s += f" {iu.p_future}"
            s += f" {iu.p_bc_a}"
            s += f" {iu.p_bc_b}"
            s += f" {iu.H}"
            s += f" {iu.v1}"
            s += f" {iu.v2}"
            s += "\n"
            self.logfile.write(s)

    def get_current_time(self):
        return time.time() - self.t0

    def prepare_run(self):
        self._thread_is_active = True
        threading.Thread(target=self._thread).start()

    def shutdown(self):
        self._thread_is_active = False
        print("Shutdown VAP")

        if self.record:
            print("Closed: ", self.filepath)
            self.logfile.close()
            self.logfile = None


if __name__ == "__main__":
    from vap_agent.modules.microphone_stereo_module import MicrophoneStereoModule
    from vap_agent.modules.audio_to_tensor_module import AudioToTensor

    mic = MicrophoneStereoModule()
    vapper = VapModule(buffer_time=20)
    a2t = AudioToTensor(buffer_time=20, device=vapper.device)
    clb = retico_core.debug.CallbackModule(vapCallback)

    mic.subscribe(a2t)
    a2t.subscribe(vapper)
    vapper.subscribe(clb)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
