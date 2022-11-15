import retico_core
import torch

from utils import TensorIU

NORM_FACTOR = 1 / (2 ** 15)


def collector_callback(update_msg):
    for x, ut in update_msg:
        print(f"{ut} payload: ", x.tensor.shape, x.tensor.mean())


class AudioCollector(retico_core.abstract.AbstractModule):
    @staticmethod
    def name():
        return "AudioStreamZMQ StereoCollector"

    @staticmethod
    def description():
        return "Collects two audio streams into 1"

    @staticmethod
    def input_ius():
        return [retico_core.audio.AudioIU]

    @staticmethod
    def output_iu():
        # return retico_core.abstract.IncrementalUnit
        return TensorIU
        # return None

    def __init__(
        self,
        buffer_time=2,
        frame_time=0.02,
        sample_rate=16_000,
        sample_width=2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sample_rate = sample_rate
        self.sample_width = sample_width

        self.chunk_size = round(sample_rate * frame_time)
        self.n_samples = int(sample_rate * buffer_time)
        self.a_tensor = torch.zeros((self.n_samples), requires_grad=False)
        self.b_tensor = torch.zeros((self.n_samples), requires_grad=False)
        print("TENSOR N_SAMPLES: ", self.n_samples)
        print("TENSOR CHUNK_SIZE: ", self.chunk_size)
        print("self.a_tensor: ", tuple(self.a_tensor.shape))
        print("self.b_tensor: ", tuple(self.b_tensor.shape))

    def update_tensor(self, audio_bytes, x):
        chunk = torch.frombuffer(
            audio_bytes, dtype=torch.int16
        )  # .float() * NORM_FACTOR

        chunk = chunk.float() * NORM_FACTOR

        # Move older samples back
        x = x.roll(-self.chunk_size, 0)

        # Update new values
        x[-self.chunk_size :] = chunk
        return x

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut == retico_core.UpdateType.ADD:
                if iu.speaker == 0:
                    self.a_tensor = self.update_tensor(iu.raw_audio, self.a_tensor)
                else:
                    self.b_tensor = self.update_tensor(iu.raw_audio, self.b_tensor)

        # print(self.a_tensor.abs().max().item(), self.b_tensor.abs().max().item())
        output_iu = self.create_iu()
        output_iu.set_tensor(torch.stack([self.a_tensor, self.b_tensor]))
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
