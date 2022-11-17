import retico_core

from vap_agent.vap_module import VapIU


def turnTakingCallback(update_msg):
    pass


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
        n_threshold_active=3,
        cli_print: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_threshold_active = n_threshold_active
        self.last_speaker = None
        self.na_active = 0
        self.nb_active = 0

        self.cli_print = cli_print

    def get_speaker_probs_text(self, pp, n=100, marker="▉"):
        p = int(n * pp)
        pn = n - p
        text = "A |"
        text += marker * p
        text += "-" * pn
        text += "| B"
        return text

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut != retico_core.UpdateType.ADD:
                continue

            if self.last_speaker is None:
                self.last_speaker = "a" if iu.p_now < 0.5 else "b"

            if self.last_speaker == "a":
                if iu.p_now > 0.5:
                    self.nb_active += 1
                    if self.nb_active >= self.n_threshold_active:
                        print("           ╰──> BBBBBBBBBBBBB")
                        self.last_speaker = "b"
                        self.na_active = 0
            else:
                if iu.p_now < 0.5:
                    self.na_active += 1
                    if self.na_active >= self.n_threshold_active:
                        print("AAAAAA <───╯ ")
                        self.last_speaker = "a"
                        self.nb_active = 0

            if self.cli_print:
                pnow_text = self.get_speaker_probs_text(iu.p_now)
                pfut_text = self.get_speaker_probs_text(iu.p_future, marker="|")
                print(pnow_text)
                print(pfut_text)


if __name__ == "__main__":
    from vap_agent.microphone_stereo_module import MicrophoneStereoModule
    from vap_agent.audio_to_tensor_module import AudioToTensor
    from vap_agent.vap_module import VapModule

    mic = MicrophoneStereoModule()
    vapper = VapModule(buffer_time=20, refresh_time=0.1)
    a2t = AudioToTensor(buffer_time=20, device=vapper.device)
    tt = TurnTakingModule(clip_print=True)

    mic.subscribe(a2t)
    a2t.subscribe(vapper)
    vapper.subscribe(tt)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
