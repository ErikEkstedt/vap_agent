from os.path import join
from os import makedirs
from dataclasses import dataclass
import retico_core
import time

from vap_agent.audio_recorder_module import AudioRecorderModule
from vap_agent.microphone_stereo_module import MicrophoneStereoModule, AudioToTensor
from vap_agent.vap_module import VapModule


@dataclass
class AgentConfig:
    savepath: str
    sample_rate: int = 16_000
    sample_width: int = 2
    sample_frame_time: float = 0.02
    record: bool = True

    audio_buffer_time: float = 20
    vap_refresh_time: float = 0.1


class Agent:
    def __init__(self, conf) -> None:
        self.conf = conf

        makedirs(self.paths["root"], exist_ok=True)
        print("Created: ", self.paths["root"])

        self.build()

    @property
    def paths(self):
        paths = {
            "root": self.conf.savepath,
            "audio": join(self.conf.savepath, "audio.wav"),
        }
        return paths

    def build(self):
        self.audio_in = MicrophoneStereoModule(
            frame_length=self.conf.sample_frame_time,
            sample_width=self.conf.sample_width,
            sample_rate=self.conf.sample_rate,
            channels=2,
        )
        self.vap = VapModule(
            buffer_time=self.conf.audio_buffer_time,
            refresh_time=self.conf.vap_refresh_time,
        )
        self.audio_to_tensor = AudioToTensor(
            buffer_time=self.conf.audio_buffer_time, device=self.vap.device
        )

        if self.conf.record:
            self.audio_recorder = AudioRecorderModule(
                filename=self.paths["audio"],
                sample_rate=self.conf.sample_rate,
                sample_width=self.conf.sample_width,
                channels=2,
            )

        # Connect modules
        self.audio_in.subscribe(self.audio_to_tensor)

        if self.conf.record:
            self.audio_in.subscribe(self.audio_recorder)

        self.audio_to_tensor.subscribe(self.vap)

        print("Connected modules")

    def run(self):
        t = time.time()
        retico_core.network.run(self.audio_in)
        print("#" * 40)
        print("#" * 40)
        print("Running agent")
        print("Press ENTER to exit")
        print("#" * 40)
        print("#" * 40)
        input()
        retico_core.network.stop(self.audio_in)
        t = round(time.time() - t, 2)
        print(f"Process took {t}s")


if __name__ == "__main__":
    conf = AgentConfig(savepath="test")

    agent = Agent(conf)

    agent.run()
