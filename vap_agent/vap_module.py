import retico_core
import torch
import threading
import time
import zmq

from vap_agent.utils import TensorIU

# model imports
from vap.model import VAPModel
from vap.utils import everything_deterministic


everything_deterministic()
torch.manual_seed(0)


CHECKPOINT = "../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt"


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
        checkpoint: str = CHECKPOINT,
        buffer_time: float = 10,
        refresh_time: float = 0.5,
        zmq_use: bool = False,
        zmq_port=5557,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.refresh_time = refresh_time
        self.sample_rate = 16_000
        self.buffer_time = buffer_time
        self.n_samples = int(buffer_time * self.sample_rate)

        # ZMQ
        self.zmq_use = zmq_use
        self.zmq_port = zmq_port
        self.zmq_topic = "vap"

        self.load_model(self.checkpoint)
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

        probs_buffer = torch.zeros(100)
        while self._thread_is_active:
            time.sleep(self.refresh_time)
            out = self.model.output(self.x)
            # pp = out["p_all"][0, -n_frames:, 1].mean().item()
            pp = out["p"][0, -n_frames:, 1].mean().item()
            probs_buffer = probs_buffer.roll(-1, 0)
            probs_buffer[-1] = pp
            pabc = round(100 * out["p_bc"][0, -n_frames:, 0].mean().item(), 2)
            pbbc = round(100 * out["p_bc"][0, -n_frames:, 1].mean().item(), 2)
            # last_pp = out["p"][0, :-n_frames, 1].mean().item()
            # cli_text = self.get_speaker_probs_print(pp)
            # cli_t = self.get_speaker_probs_print_bins(pp, last_pp)
            # print(cli_text)

            if self.zmq_use:
                self.socket.send_string(self.zmq_topic, zmq.SNDMORE)
                self.socket.send_pyobj(probs_buffer)

    def setup(self):
        print("SETUP VAP")
        if self.zmq_use:
            socket_ip = f"tcp://*:{self.zmq_port}"
            self.socket = zmq.Context().socket(zmq.PUB)
            self.socket.bind(socket_ip)
            print("Setup VAP ZMQ socket: ", socket_ip)

    def prepare_run(self):
        self._thread_is_active = True
        threading.Thread(target=self._thread).start()

    def shutdown(self):
        self._thread_is_active = False


if __name__ == "__main__":
    from vap_agent.microphone_stereo_module import MicrophoneStereoModule
    from vap_agent.audio_to_tensor_module import AudioToTensor, audioTensorCallback

    mic = MicrophoneStereoModule()
    vapper = VapModule(buffer_time=10, refresh_time=0.1, zmq_use=True)
    a2t = AudioToTensor(buffer_time=10, device=vapper.device)
    # clb = retico_core.debug.CallbackModule(audioTensorCallback)

    mic.subscribe(a2t)
    a2t.subscribe(vapper)
    # a2t.subscribe(clb)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
