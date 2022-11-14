import retico_core
from retico_core.audio import AudioIU
import time

import torch
import threading

from vap.model import VAPModel
from vap.utils import load_sample, batch_to_device, everything_deterministic, write_json


everything_deterministic()
torch.manual_seed(0)


class VAP(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "VAP Module"

    @staticmethod
    def description():
        return "A module that collects audio from two microphone inputs."

    @staticmethod
    def input_ius():
        return [AudioIU]

    def __init__(
        self,
        checkpoint="",
        buffer_time=10,
        frame_time=0.02,
        audio_sample_rate=16_000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint = checkpoint
        self.sample_rate = 16_000
        self.buffer_time = buffer_time
        self.audio_sample_rate = audio_sample_rate
        self.audio_buffer = []

        assert (
            self.sample_rate == audio_sample_rate
        ), f"Expected audio and model to have the same sample rate but model: {self.sample_rate} and audio: {self.audio_sample_rate}"

        self.load_model(
            "../VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.ckpt"
        )
        self.chunk_size = int(frame_time * self.audio_sample_rate)
        self.n_samples = int(buffer_time * self.sample_rate)
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

    def reset(self):
        self.x = torch.zeros((1, 2, self.n_samples))

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut != retico_core.UpdateType.ADD:
                continue
            self.add_audio(iu.raw_audio)

    def update_tensor(self, chunk):
        # Normalize int16 -> -1, 1
        chunk = chunk / 16384.0

        # Add most recent chunk to the end
        # WARNING! only for a single channel update
        self.x[..., : -self.chunk_size] = self.x[..., self.chunk_size :]
        self.x[0, 0, -self.chunk_size :] = chunk.to(self.device)
        # self.x[0, :, -self.chunk_size :] = torch.stack([chunk, torch.zeros_like(chunk)])
        # When we update two channels we must do something different
        # self.x[0, 0, : -self.chunk_size] = self.x[0, 0, self.chunk_size :].clone()
        # self.x[0, 0, -self.chunk_size :] = chunk

    def add_audio(self, audio):
        chunk = torch.frombuffer(audio, dtype=torch.int16).float()
        self.update_tensor(chunk)

    def _vap_thread(self):
        while self._vap_thread_active:
            time.sleep(0.5)
            out = self.model.output(self.x)
            pp = out["p"][0, -10:, 1].mean().item()
            print("Speaker B -> ", round(100 * pp, 2))

    def prepare_run(self):
        self._vap_thread_active = True
        threading.Thread(target=self._vap_thread).start()

    def shutdown(self):
        self._vap_thread_active = False
        self.reset()


if __name__ == "__main__":
    from retico_core.audio import SpeakerModule, MicrophoneModule

    sample_rate = 16000
    mic = MicrophoneModule(rate=sample_rate)
    vapper = VAP(audio_sample_rate=sample_rate, buffer_time=10)
    speaker = SpeakerModule(rate=sample_rate)

    # Setup connections
    mic.subscribe(vapper)
    mic.subscribe(speaker)
    if True:
        print("=" * 40)
        print("run")
        print("=" * 40)
        retico_core.network.run(mic)
        input()
        retico_core.network.stop(mic)
