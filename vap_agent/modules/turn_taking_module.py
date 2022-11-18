from dataclasses import dataclass
from os.path import join
import json
import retico_core
import time
import zmq

from vap_agent.modules.vap_module import VapIU


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as jsonfile:
        json.dump(data, jsonfile, ensure_ascii=False)


def turnTakingCallback(update_msg):
    pass


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    text: str = ""


"""
Add a NEUTRAL turn  [0.4, .6]  ? A self-selection TRP
"""


class TurnTakingModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "TurnTakingModule"

    @staticmethod
    def description():
        return "Turn-taking"

    @staticmethod
    def input_ius():
        return [VapIU]

    def __init__(
        self,
        n_threshold_active=2,
        cli_print: bool = False,
        root: str = "",
        record: bool = False,
        zmq_use: bool = False,
        zmq_port: int = 5558,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_threshold_active = n_threshold_active
        self.cli_print = cli_print
        self.record = record

        # Paths
        self.root = root
        self.filepath = join(root, "turn_taking.json")
        self.log_filepath = join(root, "turn_taking_log.txt")

        # Log file
        self.log_vap = None

        # Turn state
        self.dialog = []
        self.last_speaker = None
        self.na_active, self.nb_active = 0, 0
        self.b_prev_start, self.a_prev_start = 0, 0

        # ZMQ
        self.zmq_use = zmq_use
        self.zmq_port = zmq_port
        self.zmq_topic = "new_speaker"

    def get_speaker_probs_text(self, pp, n=100, marker="▉"):
        p = int(n * pp)
        pn = n - p
        text = "A |"
        text += marker * p
        text += "-" * pn
        text += "| B"
        return text

    def get_current_time(self):
        return time.time() - self.t0

    def add_last_turn(self, speaker):

        end = self.get_current_time()
        if speaker == "a":
            start = self.a_prev_start
        else:
            start = self.b_prev_start

        turn = {"start": start, "end": end, "speaker": speaker}
        # turn = Turn(start=start, end=end, speaker=speaker)
        self.dialog.append(turn)

    def zmq_update_speaker(self, new_current_speaker):
        self.socket.send_string(self.zmq_topic, zmq.SNDMORE)
        self.socket.send_string(new_current_speaker)

    def update_turn_state(self, iu):
        if self.last_speaker is None:
            self.last_speaker = "a" if iu.p_now > 0.5 else "b"

        # TURN-SHIFT CONDITIONS
        if self.last_speaker == "a":

            if iu.p_now < 0.5:
                self.nb_active += 1
                if self.nb_active >= self.n_threshold_active:

                    # We add previous turn (forced to belong to a)
                    self.add_last_turn(speaker="a")

                    # Take Turn as B
                    if self.zmq_use:
                        self.zmq_update_speaker("b")
                    print("           ╰──> BBBBBBBBBBBBB")
                    self.last_speaker = "b"
                    self.na_active = 0
                    self.b_prev_start = self.get_current_time()
        else:
            if iu.p_now > 0.5:
                self.na_active += 1
                if self.na_active >= self.n_threshold_active:

                    # We add previous turn (forced to belong to b)
                    self.add_last_turn(speaker="b")

                    # Take Turn as A
                    if self.zmq_use:
                        self.zmq_update_speaker("a")
                    print("AAAAAA <───╯ ")
                    self.last_speaker = "a"
                    self.nb_active = 0
                    self.a_prev_start = self.get_current_time()

    def log(self, iu):
        """
        log IU information
        TIME P-NOW P-FUTURE P-BC-A P-BC-B
        """
        if self.log_vap:
            t = round(self.get_current_time(), 3)
            s = f"{t}"
            s += f" {iu.p_now}"
            s += f" {iu.p_future}"
            s += f" {iu.p_bc_a}"
            s += f" {iu.p_bc_b}"
            s += "\n"
            self.log_vap.write(s)

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut != retico_core.UpdateType.ADD:
                continue

            self.update_turn_state(iu)
            self.log(iu)

            if self.cli_print:
                pnow_text = self.get_speaker_probs_text(iu.p_now)
                pfut_text = self.get_speaker_probs_text(iu.p_future, marker="=")
                print(pnow_text)
                print(pfut_text)

    def setup(self):
        self.t0 = time.time()
        if self.record:
            self.log_vap = open(self.log_filepath, "w")
            self.log_vap.write("TIME P-NOW P-FUTURE P-BC-A P-BC-B\n")

        if self.zmq_use:
            pass
            socket_ip = f"tcp://*:{self.zmq_port}"
            self.socket = zmq.Context().socket(zmq.PUB)
            self.socket.bind(socket_ip)
            print("Setup TurnTaking ZMQ socket: ", socket_ip)

    def shutdown(self):

        # last turn
        if self.last_speaker != self.dialog[-1]["speaker"]:
            self.add_last_turn(self.last_speaker)

        write_json(self.dialog, self.filepath)
        if self.log_vap:
            self.log_vap.close()
            self.log_vap = None


if __name__ == "__main__":
    from vap_agent.modules.microphone_stereo_module import MicrophoneStereoModule
    from vap_agent.modules.audio_to_tensor_module import AudioToTensor
    from vap_agent.modules.vap_module import VapModule

    mic = MicrophoneStereoModule()
    vapper = VapModule(buffer_time=20, refresh_time=0.1)
    a2t = AudioToTensor(buffer_time=20, device=vapper.device)
    tt = TurnTakingModule(cli_print=True, record=False)

    mic.subscribe(a2t)
    a2t.subscribe(vapper)
    vapper.subscribe(tt)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
