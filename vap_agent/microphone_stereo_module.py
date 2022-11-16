import retico_core
import torch
import queue
import pyaudio


NORM_FACTOR = 1 / (2 ** 15)


def torchify_audio_callback(update_msg):
    for iu, ut in update_msg:
        if ut != retico_core.UpdateType.ADD:
            continue
        chunk = torch.frombuffer(iu.raw_audio, dtype=torch.int16).float() * NORM_FACTOR
        b = chunk[::2]
        a = chunk[1::2]
        print(a.max(), " <----> ", b.max())


class MicrophoneStereoModule(retico_core.AbstractProducingModule):
    """A module that produces IUs containing audio signals that are captures by
    a microphone."""

    @staticmethod
    def name():
        return "Microphone Module"

    @staticmethod
    def description():
        return "A prodicing module that records audio from microphone."

    @staticmethod
    def output_iu():
        return retico_core.audio.AudioIU

    def __init__(
        self,
        frame_length=0.02,
        sample_width=2,
        sample_rate=16_000,
        channels=2,
        device=None,
        **kwargs,
    ):
        """Initialize the Microphone Module. Args:
        frame_length (float): The length of one frame (i.e., IU) in seconds
        rate (int): The frame rate of the recording
        sample_width (int): The width of a single sample of audio in bytes.
        """
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.sample_width = sample_width
        self.sample_rate = sample_rate
        self.channels = channels

        # Pyaudio
        self._p = pyaudio.PyAudio()
        self.audio_buffer = queue.Queue()
        self.stream = None

        # Get correct device
        self.device = device
        self.device_index = self.get_device_index(device)
        self.device_info = self._p.get_device_info_by_index(self.device_index)
        # self.rate = int(self.device_info["defaultSampleRate"])
        self.chunk_size = round(self.sample_rate * frame_length)

    def __repr__(self):
        s = "MicrophoneModule"
        s += f"\nSampleRate: {self.sample_rate}"
        s += f"\nFrame length: {self.frame_length}"
        s += f"\nSample width: {self.sample_width}"
        s += f"\nChannels: {self.channels}"
        s += f"\nDevice: {self.device}"
        s += f"\nDevice index: {self.device_index}"
        return s

    def get_device_index(self, device):
        if device is None:
            info = self._p.get_default_input_device_info()
            return int(info["index"])

        bypass_index = -1
        for i in range(self._p.get_device_count()):
            info = self._p.get_device_info_by_index(i)
            if info["name"] == device:
                bypass_index = i
                break
        return bypass_index

    def callback(self, in_data, frame_count, time_info, status):
        """The callback function that gets called by pyaudio.
        Args:
            in_data (bytes[]): The raw audio that is coming in from the
                microphone
            frame_count (int): The number of frames that are stored in in_data
        """
        self.audio_buffer.put(in_data)
        return (in_data, pyaudio.paContinue)

    def process_update(self, _):
        if not self.audio_buffer:
            return None
        try:
            sample = self.audio_buffer.get(timeout=1.0)
        except queue.Empty:
            return None
        output_iu = self.create_iu()
        output_iu.set_audio(
            sample, self.chunk_size, self.sample_rate, self.sample_width
        )
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

    def setup(self):
        """Set up the microphone for recording."""

        print("Setup Microphone")
        self.stream = self._p.open(
            format=self._p.get_format_from_width(self.sample_width),
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            output=False,
            stream_callback=self.callback,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index,
            start=False,
        )
        if self.device is None:
            device = self._p.get_default_input_device_info()
            print(f"Device: {device['name']}")
        else:
            print(f"Device: {self.device}")

    def prepare_run(self):
        if self.stream:
            self.stream.start_stream()

    def shutdown(self):
        """Close the audio stream."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.stream = None
        self.audio_buffer = queue.Queue()


if __name__ == "__main__":
    mic = MicrophoneStereoModule()
    clb = retico_core.debug.CallbackModule(torchify_audio_callback)
    mic.subscribe(clb)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
