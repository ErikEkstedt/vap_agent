import retico_core
import torch

from vap_agent.utils import TensorIU

NORM_FACTOR = 1 / (2 ** 15)


def audioTensorCallback(update_msg):
    for iu, ut in update_msg:
        if ut != retico_core.UpdateType.ADD:
            continue
        t = iu.tensor
        print(t[0, 0].max(), " <----> ", t[0, 1].max())


class AudioToTensor(retico_core.AbstractModule):
    """A module that produces IUs containing audio signals that are captures by
    a microphone."""

    @staticmethod
    def name():
        return "Microphone Module"

    @staticmethod
    def description():
        return "A prodicing module that records audio from microphone."

    @staticmethod
    def input_ius():
        return [retico_core.audio.AudioIU]

    @staticmethod
    def output_iu():
        return TensorIU

    def __init__(self, buffer_time=2, sample_rate=16_000, device="cpu", **kwargs):
        """Initialize the Microphone Module. Args:
        frame_length (float): The length of one frame (i.e., IU) in seconds
        rate (int): The frame rate of the recording
        sample_width (int): The width of a single sample of audio in bytes.
        """
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.buffer_time = buffer_time
        self.device = device

        self.n_samples = int(sample_rate * buffer_time)
        self.tensor = torch.zeros((1, 2, self.n_samples), device=device)

    def update_tensor(self, audio_bytes):
        chunk = torch.frombuffer(audio_bytes, dtype=torch.int16).float() * NORM_FACTOR

        # Split stereo audio
        a = chunk[::2]
        b = chunk[1::2]
        chunk_size = a.shape[0]

        # Move values back
        self.tensor = self.tensor.roll(-chunk_size, -1)
        self.tensor[0, 0, -chunk_size:] = a.to(self.device)
        self.tensor[0, 1, -chunk_size:] = b.to(self.device)

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut == retico_core.UpdateType.ADD:
                self.update_tensor(iu.raw_audio)

        output_iu = self.create_iu()
        output_iu.set_tensor(self.tensor)
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

    def setup(self):
        pass

    def prepare_run(self):
        pass

    def shutdown(self):
        """Close the audio stream."""
        self.tensor = torch.zeros((1, 2, self.n_samples), device=self.device)


if __name__ == "__main__":

    from vap_agent.modules.microphone_stereo_module import MicrophoneStereoModule

    mic = MicrophoneStereoModule()
    a2t = AudioToTensor()
    clb = retico_core.debug.CallbackModule(audioTensorCallback)
    mic.subscribe(a2t)
    a2t.subscribe(clb)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
