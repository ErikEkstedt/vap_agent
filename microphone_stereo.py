import retico_core
from retico_core.audio import AudioIU

import torch
import numpy as np
import threading
import queue
import pyaudio

"""
https://github.com/retico-team/retico-core
https://github.com/retico-team/retico-wav2vecasr
https://github.com/retico-team/retico

https://github.com/ErikEkstedt/retico/blob/master/retico/agent/hearing.py
"""


def get_name_to_index(p=None):
    if p is None:
        p = pyaudio.PyAudio()
    name2idx = {}
    idx2name = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name2idx[info["name"]] = i
        idx2name.append(info["name"])
    return name2idx, idx2name


class MicrophoneModuleDevice(retico_core.AbstractProducingModule):
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
        return AudioIU

    def __init__(
        self,
        speaker=0,
        frame_length=0.02,
        sample_width=2,
        channels=1,
        device=None,
        # rate=44100,
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
        self.channels = channels
        self.speaker = speaker

        # Pyaudio
        self._p = pyaudio.PyAudio()
        self.audio_buffer = queue.Queue()
        self.stream = None

        # Get correct device
        self.device = device
        self.device_index = self.get_device_index(device)
        self.device_info = self._p.get_device_info_by_index(self.device_index)
        self.rate = int(self.device_info["defaultSampleRate"])
        self.chunk_size = round(self.rate * frame_length)

    def __repr__(self):
        s = "MicrophoneModule"
        s += f"\nDevice: {self.device}"
        s += f"\nRate: {self.rate}"
        s += f"\nFrame length: {self.frame_length}"
        s += f"\nSample width: {self.sample_width}"
        s += f"\nChannels: {self.channels}"
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
        output_iu.set_audio(sample, self.chunk_size, self.rate, self.sample_width)
        # Set what speaker I am
        # output_iu.speaker = self.speaker
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

    def setup(self):
        """Set up the microphone for recording."""
        self.stream = self._p.open(
            format=self._p.get_format_from_width(self.sample_width),
            channels=self.channels,
            rate=self.rate,
            input=True,
            output=False,
            stream_callback=self.callback,
            frames_per_buffer=self.chunk_size,
            input_device_index=self.device_index,
            start=False,
        )

        print("Setup Microphone")
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
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
        self.audio_buffer = queue.Queue()


# TODO: callback module to show AudioIO inputs from separate channels/speakers


def debug():
    def callback(in_data, frame_count, time_info, status):
        """The callback function that gets called by pyaudio.
        Args:
            in_data (bytes[]): The raw audio that is coming in from the
                microphone
            frame_count (int): The number of frames that are stored in in_data
        """
        print("frame_count: ", frame_count)
        return (in_data, pyaudio.paContinue)

    p = pyaudio.PyAudio()

    n2i, i2n = get_name_to_index(p)

    frame_length = 0.02
    sample_width = 2
    channels = 1
    rate = 48000
    chunk_size = round(rate * frame_length)
    device_index = 31
    info = p.get_device_info_by_index(device_index)
    api_info = p.get_device_info_by_host_api_device_index(info["hostApi"], 1)

    print("Name: ", i2n[device_index])
    for k, v in info.items():
        print(f"{k}: {v}")

    stream = p.open(
        format=p.get_format_from_width(sample_width),
        channels=channels,
        rate=rate,
        input=True,
        output=False,
        stream_callback=callback,
        frames_per_buffer=chunk_size,
        input_device_index=device_index,
        start=False,
    )

    n2i = get_name_to_index()


if __name__ == "__main__":
    from retico_core.network import run, stop
    from retico_core.audio import SpeakerModule, MicrophoneModule
    from retico_core.debug import CallbackModule

    n2i = get_name_to_index()

    # device = "Sony SingStar USBMIC Analog Stereo"
    # device = "SteelSeries Arctis 7 Chat"
    device = None
    sample_rate = 16000
    mic = MicrophoneModule(rate=sample_rate)
    print("-" * 40)
    print(mic)
    print("-" * 40)
    mic.setup()

    # device1 = "SteelSeries Arctis 7 Chat"
    # mic = MicrophoneModule(device=device1, speaker=1, rate=48000)
    # mic = MicrophoneModule()
    # print(mic)
    # print("-" * 40)

    if True:
        vv = VAP(audio_sample_rate=sample_rate, buffer_time=2)
        speaker = SpeakerModule(rate=sample_rate)
        mic.subscribe(vv)
        mic.subscribe(speaker)

        # mic1.subscribe(vv)
        # clb = CallbackModule(mic_callback)
        # setup
        # mic.subscribe(clb)
        # mic1.subscribe(speaker)
        print("run")
        run(mic)
        # run(mic1)
        input()
        stop(mic)
        # stop(mic1)
