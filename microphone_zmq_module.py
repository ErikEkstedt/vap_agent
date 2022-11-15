import retico_core
import zmq

ctx = zmq.Context(io_threads=2)

"""

Microphone nodes runs `audio_zmq.py` with relevant ports and IPs
* `python audio_zmq.py -p 5555 --ip localhost`
* `python audio_zmq.py -p 5556 --ip 192.168.0.104`

Run this retico program
* `python zmq_microphone_in.py`
"""


def audio_callback(update_msg):
    for x, ut in update_msg:
        print(f"{ut} {x.speaker} audio: ", len(x.raw_audio))


class MicrophoneZMQModule(retico_core.abstract.AbstractProducingModule):
    @staticmethod
    def name():
        return "AudioStreamZMQ"

    @staticmethod
    def description():
        return "Listens to a microphone throuch ZMQ"

    @staticmethod
    def output_iu():
        return retico_core.audio.AudioIU

    def __init__(
        self,
        port,
        speaker=0,
        frame_time=0.02,
        sample_rate=16_000,
        sample_width=2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.port = port
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.chunk_size = round(sample_rate * frame_time)
        self.socket = None

    def process_update(self, _):
        sample = self.socket.recv()
        output_iu = self.create_iu()
        output_iu.set_audio(
            sample, self.chunk_size, self.sample_rate, self.sample_width
        )
        output_iu.speaker = self.speaker
        return retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)

    def setup(self):
        pass

    def prepare_run(self):
        self.socket = ctx.socket(zmq.PAIR)
        print("Created AudioStreamZMQ socket")
        # self.socket = ctx.socket(zmq.SUB)
        # print(f"tcp://*:{self.port}")
        addr = f"tcp://*:{self.port}"
        self.socket.bind(addr)
        print("Bind socket -> ", addr)

    def shutdown(self):
        self.socket.close()
        self.socket = None
        print("Closed socket")


if __name__ == "__main__":

    from audio_collector_module import AudioCollector, collector_callback

    mic = MicrophoneZMQModule(speaker=0, port=5555)
    mic2 = MicrophoneZMQModule(speaker=1, port=5556)
    coll = AudioCollector()
    mic.subscribe(coll)
    mic2.subscribe(coll)

    # clb = retico_core.debug.CallbackModule(audio_callback)
    # mic.subscribe(clb)
    # mic2.subscribe(clb)

    clb = retico_core.debug.CallbackModule(collector_callback)
    coll.subscribe(clb)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
