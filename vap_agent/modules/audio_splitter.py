import retico_core
import numpy as np


def audioSplitterCallback(update_msg):
    for iu, ut in update_msg:
        if ut != retico_core.UpdateType.ADD:
            continue
        print(iu.speaker, len(iu.raw_audio))


class AudioSplitterModule(retico_core.AbstractModule):
    """A Module that consumes AudioIUs and saves them as a PCM wave file to
    disk."""

    @staticmethod
    def name():
        return "Audio Recorder Module"

    @staticmethod
    def description():
        return "A Module that saves incoming audio to disk."

    @staticmethod
    def input_ius():
        return [retico_core.audio.AudioIU]

    @staticmethod
    def output_iu():
        return retico_core.audio.AudioIU

    def __init__(
        self,
        speaker="a",
        frame_length=0.02,
        sample_rate=16_000,
        sample_width=2,
        **kwargs,
    ):
        """Initialize the audio recorder module.
        Args:
            filename (string): The file name where the audio should be recorded
                to. The path to the file has to be created beforehand.
            rate (int): The sample rate of the input and thus of the wave file.
                Defaults to 16000.
            sample_width (int): The width of one sample. Defaults to 2.
        """
        super().__init__(**kwargs)
        self.speaker = speaker
        self.frame_length = frame_length
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.chunk_size = round(self.sample_rate * frame_length)

        self.offset = 0
        if speaker == "b":
            self.offset = 1

    def split_audio(self, audio_bytes):
        chunk = np.frombuffer(audio_bytes, dtype=np.int16)
        a = chunk[self.offset :: self.sample_width].tobytes()
        # b = chunk[1 :: self.sample_width].tobytes()

        # Can this be done faster on the bytes directly?
        # the below only gives NOISE be careful
        # a = audio_bytes[:: 2 * self.sample_width]
        # b = audio_bytes[1 :: 2 * self.sample_width]
        return a

    def process_update(self, update_msg):
        um = retico_core.UpdateMessage()
        for iu, ut in update_msg:
            if ut != retico_core.UpdateType.ADD:
                continue

            a = self.split_audio(iu.raw_audio)
            a_output_iu = self.create_iu(iu)
            a_output_iu.set_audio(
                a, self.chunk_size, self.sample_rate, self.sample_width
            )
            a_output_iu.speaker = "a"
            um.add_iu(a_output_iu, retico_core.UpdateType.ADD)

            # b_output_iu = self.create_iu(iu)
            # b_output_iu.set_audio(
            #     b, self.chunk_size, self.sample_rate, self.sample_width
            # )
            # b_output_iu.speaker = "b"
            # um.add_iu(b_output_iu, retico_core.UpdateType.ADD)
            #
        if len(um) > 0:
            return um


if __name__ == "__main__":

    from vap_agent.modules import MicrophoneStereoModule

    mic = MicrophoneStereoModule()
    split = AudioSplitterModule(speaker="b")
    speaker = retico_core.audio.SpeakerModule(rate=16000)

    clb = retico_core.debug.CallbackModule(audioSplitterCallback)
    mic.subscribe(split)
    split.subscribe(clb)
    split.subscribe(speaker)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
