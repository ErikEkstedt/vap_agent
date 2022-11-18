from os.path import join
from pathlib import Path
import datetime

from dataclasses import dataclass, asdict
import retico_core
import time

from vap_agent.modules import (
    AudioRecorderModule,
    AudioRecorderModule,
    MicrophoneStereoModule,
    AudioToTensor,
    TurnTakingModule,
    VapModule,
)
from vap.utils import write_json


@dataclass
class AgentConfig:
    savepath: str  # path of this particular run
    runs_path: str = "runs"  # root for all runs
    record: bool = True  # if data should be saved

    # AUDIO
    sample_rate: int = 16_000
    sample_width: int = 2
    sample_frame_time: float = 0.02

    # VAP
    audio_buffer_time: float = 20  # input duration to VAP
    vap_refresh_time: float = -0.5  # negative value processes without timeout

    # TurnTaking
    # consecutive frames of the next-speaker to trigger turn
    tt_threshold_active: int = 3
    zmq_use: bool = True
    zmq_port: int = 5558


class Agent:
    def __init__(self, conf) -> None:
        self.conf = conf

        ###############################################
        # Create savepath
        if self.conf.record:
            self.paths = self._paths()
            Path(self.paths["root"]).mkdir(parents=True, exist_ok=True)
            print("Created: ", self.paths["root"])
            self.save_conf()
        self.build()

    def _paths(self):
        datetime_now = datetime.datetime.now()
        date_name = datetime_now.strftime("%y%m%d_%H-%M")
        root = join(self.conf.runs_path, self.conf.savepath, date_name)
        paths = {
            "root": root,
            "conf": join(root, "conf.json"),
            "audio": join(root, "audio.wav"),
            "dialog": join(root, "dialog.json"),
            "turn_taking": join(root, "turn_taking.json"),
        }
        return paths

    def save_conf(self):
        write_json(asdict(self.conf), self.paths["conf"])

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
        self.turn_taking = TurnTakingModule(
            n_threshold_active=self.conf.tt_threshold_active,
            root=self.paths["root"],
            record=self.conf.record,
            zmq_use=self.conf.zmq_use,
            zmq_port=self.conf.zmq_port,
        )

        if self.conf.record:
            self.audio_recorder = AudioRecorderModule(
                filename=self.paths["audio"],
                sample_rate=self.conf.sample_rate,
                sample_width=self.conf.sample_width,
                channels=2,
            )

        print()
        print("-" * 50)
        # Connect Input modeles
        self.audio_in.subscribe(self.audio_to_tensor)
        print("Audio in      -> AudioToTensor")

        self.audio_to_tensor.subscribe(self.vap)
        print("AudioToTensor -> Vap")

        if self.conf.record:
            self.audio_in.subscribe(self.audio_recorder)
            print("Audio in      -> AudioRecord")

        # Connect acoustic vap turn-taking probs to turn-taking module
        self.vap.subscribe(self.turn_taking)
        print("VAP           -> TurnTaking")

        print("-" * 50)

    def run(self):
        t = time.time()
        retico_core.network.run(self.audio_in)
        print("Running agent")
        print("Press ENTER to exit")
        print("-" * 40)
        input()
        retico_core.network.stop(self.audio_in)
        t = round(time.time() - t, 2)
        print(f"Process took {t}s")
        print("Session saved -> ", self.paths["root"])


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Record dialog")
    args = parser.parse_args()

    conf = AgentConfig(savepath="test", record=args.record)
    agent = Agent(conf)

    agent.run()
