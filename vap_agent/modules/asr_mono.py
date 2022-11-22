import threading
import retico_core
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import transformers
import pydub
import webrtcvad
import time

transformers.logging.set_verbosity_error()

"""
This module is a slight variation of the original:

    https://github.com/retico-team/retico-wav2vecasr


...with slight renaming of variables, faster refresh-time, cuda compatible, etc.
"""


NORM_FACTOR = 1 / (2 ** 15)

msg = []


def callback(update_msg):
    global msg
    for x, ut in update_msg:
        if ut == retico_core.UpdateType.ADD:
            msg.append(x)
        if ut == retico_core.UpdateType.REVOKE:
            msg.remove(x)
    txt = ""
    committed = False
    for x in msg:
        txt += x.text + " "
        committed = committed or x.committed
    if committed:
        print(" " * 80, end="\r")
        print(f"{txt}", end="\r")
        print("\n--------committed----------\n")
        msg = []


class Wav2Vec2ASR:
    def __init__(
        self,
        wav2vec2_model: str = "facebook/wav2vec2-base-960h",
        sample_rate: int = 16_000,
        silence_dur: float = 1,
        vad_agressiveness: int = 3,
        silence_threshold: float = 0.75,
    ):
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
        self.model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to("cuda")
        print("Wav2Vec2 DEVICE: ", self.device)
        self.model.freeze_feature_encoder()
        self.audio_buffer = []
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_frames = None
        self.silence_threshold = silence_threshold

    def _resample_audio(self, audio):
        if self.sample_rate != 16_000:
            # If the framerate is not 16 kHz, we need to resample
            s = pydub.AudioSegment(
                audio, sample_width=2, channels=1, frame_rate=self.sample_rate
            )
            s = s.set_frame_rate(16_000)
            return s._data
        return audio

    def get_n_sil_frames(self):
        if not self._n_sil_frames:
            if len(self.audio_buffer) == 0:
                return None
            frame_length = len(self.audio_buffer[0]) / 2
            self._n_sil_frames = int(self.silence_dur / (frame_length / 16_000))
        return self._n_sil_frames

    def recognize_silence(self):
        n_sil_frames = self.get_n_sil_frames()
        if not n_sil_frames or len(self.audio_buffer) < n_sil_frames:
            return True
        silence_counter = 0
        for a in self.audio_buffer[-n_sil_frames:]:
            if not self.vad.is_speech(a, 16_000):
                silence_counter += 1
        if silence_counter >= int(self.silence_threshold * n_sil_frames):
            return True
        return False

    def add_audio(self, audio):
        audio = self._resample_audio(audio)
        self.audio_buffer.append(audio)

    def recognize(self):
        silence = self.recognize_silence()

        if not self.vad_state and not silence:
            self.vad_state = True
            self.audio_buffer = self.audio_buffer[-self.get_n_sil_frames() :]

        if not self.vad_state:
            return None, False

        full_audio = b""
        for a in self.audio_buffer:
            full_audio += a
        # npa = np.frombuffer(full_audio, dtype=np.int16).astype(np.double)
        # if len(npa) < 10:
        #     return None, False
        # input_values = self.processor(
        #     npa, return_tensors="pt", sampling_rate=16000
        # ).input_values

        input_values = torch.frombuffer(full_audio, dtype=torch.int16)
        if len(input_values) < 10:
            return None, False

        input_values = (input_values.float() * NORM_FACTOR).view(1, -1).to(self.device)
        logits = self.model(input_values).logits
        # predicted_ids = np.argmax(logits.detach().numpy(), axis=-1)
        predicted_ids = logits.argmax(dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].lower()

        if silence:
            self.vad_state = False
            self.audio_buffer = []

        return transcription, self.vad_state

    def reset(self):
        self.vad_state = True
        self.audio_buffer = []


class Wav2VecASRModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "Wav2Vec ASR Module"

    @staticmethod
    def description():
        return "A module that recognizes speech using Wav2Vec."

    @staticmethod
    def input_ius():
        return [retico_core.audio.AudioIU]

    @staticmethod
    def output_iu():
        return retico_core.text.SpeechRecognitionIU

    LANGUAGE_MAPPING = {
        "en": "facebook/wav2vec2-base-960h",
        "de": "oliverguhr/wav2vec2-large-xlsr-53-german-cv9",
        "fr": "facebook/wav2vec2-large-xlsr-53-french",
        "es": "facebook/wav2vec2-large-xlsr-53-spanish",
    }

    def __init__(
        self,
        language="en",
        sample_rate=16000,
        silence_dur: float = 1,
        vad_agressiveness: int = 3,
        silence_threshold: float = 0.75,
        refresh_time: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if language not in self.LANGUAGE_MAPPING.keys():
            print("Unknown ASR language. Defaulting to English (en).")
            language = "en"

        self.language = language
        self.acr = Wav2Vec2ASR(
            wav2vec2_model=self.LANGUAGE_MAPPING[language],
            silence_dur=silence_dur,
            vad_agressiveness=vad_agressiveness,
            silence_threshold=silence_threshold,
        )
        self.refresh_time = refresh_time
        self.sample_rate = sample_rate
        self.silence_dur = silence_dur
        self._asr_thread_active = False
        self.latest_input_iu = None

    def process_update(self, update_message):
        for iu, ut in update_message:
            # Audio IUs are only added and never updated.
            if ut != retico_core.UpdateType.ADD:
                continue
            if self.sample_rate is None:
                self.sample_rate = iu.rate
                self.acr.sample_rate = self.sample_rate
            self.acr.add_audio(iu.raw_audio)
            if not self.latest_input_iu:
                self.latest_input_iu = iu

    def _asr_thread(self):
        while self._asr_thread_active:
            time.sleep(self.refresh_time)
            if not self.sample_rate:
                continue
            prediction, vad = self.acr.recognize()
            if prediction is None:
                continue
            end_of_utterance = not vad
            um, new_tokens = retico_core.text.get_text_increment(self, prediction)

            if len(new_tokens) == 0 and vad:
                continue

            for i, token in enumerate(new_tokens):
                output_iu = self.create_iu(self.latest_input_iu)
                eou = i == len(new_tokens) - 1 and end_of_utterance
                output_iu.set_asr_results([prediction], token, 0.0, 0.99, eou)
                self.current_output.append(output_iu)
                um.add_iu(output_iu, retico_core.UpdateType.ADD)

            if end_of_utterance:
                for iu in self.current_output:
                    self.commit(iu)
                    um.add_iu(iu, retico_core.UpdateType.COMMIT)
                self.current_output = []

            self.latest_input_iu = None
            self.append(um)

    def prepare_run(self):
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()

    def shutdown(self):
        self._asr_thread_active = False
        self.acr.reset()


if __name__ == "__main__":

    from vap_agent.modules.microphone_stereo_module import MicrophoneStereoModule
    from vap_agent.modules.audio_splitter import AudioSplitterModule

    mic = MicrophoneStereoModule()
    split = AudioSplitterModule(speaker="a")
    asr = Wav2VecASRModule(
        "en",
        sample_rate=16000,
        silence_dur=0.3,
        vad_agressiveness=3,
        silence_threshold=0.3,
        refresh_time=0.15,
    )
    clb = retico_core.debug.CallbackModule(callback=callback)

    mic.subscribe(split)
    split.subscribe(asr)
    asr.subscribe(clb)

    retico_core.network.run(mic)
    print("Running the ASR. Press enter to exit")
    input()
    retico_core.network.stop(mic)
