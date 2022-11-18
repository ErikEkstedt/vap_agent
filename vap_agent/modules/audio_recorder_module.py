import retico_core
import wave


class AudioRecorderModule(retico_core.AbstractConsumingModule):
    """A Module that consumes AudioIUs and saves them as a PCM wave file to
    disk."""

    wavfile: wave.Wave_write

    @staticmethod
    def name():
        return "Audio Recorder Module"

    @staticmethod
    def description():
        return "A Module that saves incoming audio to disk."

    @staticmethod
    def input_ius():
        return [retico_core.audio.AudioIU]

    def __init__(
        self, filename, channels=2, sample_rate=16_000, sample_width=2, **kwargs
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
        self.filename = filename
        self.channels = channels
        self.sample_rate = sample_rate
        self.sample_width = sample_width

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.wavfile.writeframes(iu.raw_audio)

    def setup(self):
        self.wavfile = wave.open(self.filename, "wb")
        self.wavfile.setnchannels(self.channels)
        self.wavfile.setsampwidth(self.sample_width)
        self.wavfile.setframerate(self.sample_rate)

    def shutdown(self):
        self.wavfile.close()
