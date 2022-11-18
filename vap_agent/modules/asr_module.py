import retico_core
import threading
import time
import torch
from typing import List
from os.path import join

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from vap_agent.utils import TensorIU


"""
Heavily inspired by and copied from:
https://github.com/retico-team/retico-wav2vecasr/blob/main/retico_wav2vecasr/wav2vecasr.py
"""


def asrCallback(update_msg):
    for iu, ut in update_msg:
        # if ut != retico_core.UpdateType.ADD:
        #     continue
        print(iu)
        print()


class Wav2VecASR:
    def __init__(self, wav2vec2_model="facebook/wav2vec2-base-960h"):
        self.device = "cpu"

        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
        self.model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
        self.model.freeze_feature_encoder()
        self.model.eval()

        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to("cuda")

    def recognize(self, x):
        logits = self.model(x).logits
        predicted_ids = logits.argmax(dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].lower()
        return transcription


class ASRModule(retico_core.AbstractModule):
    MODEL_NAMES: List[str] = ["wav2vec"]
    LOG_HEAD: str = "TIME WORD"

    @staticmethod
    def name():
        return "PyTorch ASR Module"

    @staticmethod
    def description():
        return "An ASR"

    @staticmethod
    def input_ius():
        return [TensorIU]

    @staticmethod
    def output_iu():
        return retico_core.text.SpeechRecognitionIU

    def __init__(
        self,
        model_name: str = "wav2vec2",
        refresh_time=0.1,
        record=False,
        root="",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.sample_rate = 16_000
        self.refresh_time = refresh_time

        self.load_asr(model_name)

        self.buffer_time = 5
        self.buffer_n_samples = int(self.buffer_time * self.sample_rate)
        self.tensor = None

        # PAths
        self.record = record
        self.root = root
        self.filepath = join(root, "asr.txt")
        self.latest_input_iu = None
        self.last_transcription = ""

    @property
    def device(self):
        return self.asr.device

    def load_asr(self, model_name):
        print()
        print("-" * 50)
        print("Load Wav2VecASR Model")
        if model_name == "wav2vec2":
            self.asr = Wav2VecASR()
        else:
            raise NotImplementedError(
                f"{model_name} not implemented. try {self.MODEL_NAMES}"
            )
        print("Device: ", self.device)
        print("-" * 50)
        print()

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut != retico_core.UpdateType.ADD:
                continue
            # update current tensor
            self.tensor = iu.tensor
            if not self.latest_input_iu:
                self.latest_input_iu = iu

    def _thread(self):
        while self._thread_is_active:
            if self.refresh_time > 0:
                time.sleep(self.refresh_time)
            if self.tensor is None:
                continue

            # Speaker A
            transcription = self.asr.recognize(
                self.tensor[:, 0, -self.buffer_n_samples :]
            )
            if transcription == self.last_transcription:
                continue

            self.last_transcription = transcription

            um, new_tokens = retico_core.text.get_text_increment(self, transcription)

            if len(new_tokens) == 0:
                continue

            # self.log_transcription(transcription)

            # print(t, transcription)

            for i, token in enumerate(new_tokens):
                output_iu = self.create_iu()
                output_iu.set_asr_results(
                    [transcription], token, 0.0, 0.99, final=False
                )
                self.current_output.append(output_iu)
                um.add_iu(output_iu, retico_core.UpdateType.ADD)
                # t = round(time.time() - self.t0, 3)

            # if len(self.current_output) > 5:
            #     for iu in self.current_output:
            #         self.commit(iu)
            #         um.add_iu(iu, retico_core.UpdateType.COMMIT)
            #     # print(um)
            #     self.current_output = []

            # print(len(self.current_output))
            self.log(output_iu)
            self.append(um)

    def get_current_time(self):
        return time.time() - self.t0

    def log(self, iu):
        """
        log IU information
        """

        if self.logfile is not None:
            t = round(self.get_current_time(), 3)
            s = f"{t}"
            s += f" {iu.payload}"
            s += "\n"
            self.logfile.write(s)

    def log_transcription(self, transcription: str):
        """
        log transcription directly
        """

        if self.logfile is not None:
            t = round(self.get_current_time(), 3)
            s = f"{t}"
            s += f" {transcription}"
            s += "\n"
            self.logfile.write(s)

    def prepare_run(self):
        self._thread_is_active = True
        threading.Thread(target=self._thread).start()

    def setup(self):
        self.t0 = time.time()

        if self.record:
            # add first row of txt-file
            self.logfile = open(self.filepath, "w")
            self.logfile.write(self.LOG_HEAD + "\n")

    def shutdown(self):
        self._thread_is_active = False

        if self.record:
            self.logfile.close()
            self.logfile = None
            print("Closed: ", self.filepath)


if __name__ == "__main__":
    from vap_agent.modules.microphone_stereo_module import MicrophoneStereoModule
    from vap_agent.modules.audio_to_tensor_module import AudioToTensor

    mic = MicrophoneStereoModule()
    asr = ASRModule()
    a2t = AudioToTensor(buffer_time=20, device=asr.device)
    clb = retico_core.debug.CallbackModule(asrCallback)

    mic.subscribe(a2t)
    a2t.subscribe(asr)
    asr.subscribe(clb)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
