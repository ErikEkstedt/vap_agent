import zmq
import retico_core
import threading
import time

from vap_agent.utils import TensorIU


class TensorZMQModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "ZeroMQ Tensor Module"

    @staticmethod
    def description():
        return "A Module providing writing onto a ZeroMQ bus"

    @staticmethod
    def input_ius():
        return [TensorIU]

    def __init__(
        self,
        topic: str,
        port: int = 5557,
        refresh_time: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topic = topic
        self.port = port
        self.refresh_time = refresh_time
        self.tensor = None

    def process_update(self, update_msg):
        for iu, ut in update_msg:
            if ut != retico_core.UpdateType.ADD:
                continue
            self.tensor = iu.tensor

    def setup(self):
        socket_ip = f"tcp://*:{self.port}"
        self.socket = zmq.Context().socket(zmq.PUB)
        self.socket.bind(socket_ip)
        print("Setup TensorZMQModule socket: ", socket_ip)

    def prepare_run(self):
        self._thread_is_active = True
        threading.Thread(target=self._thread).start()

    def _thread(self):
        while self._thread_is_active:
            time.sleep(self.refresh_time)
            if self.tensor is not None:
                self.socket.send_string(self.topic, zmq.SNDMORE)
                self.socket.send_pyobj(self.tensor.cpu())

    def shutdown(self):
        print("Shutdown TensorZMQModule")
        self._thread_is_active = False


if __name__ == "__main__":
    from vap_agent.microphone_stereo_module import MicrophoneStereoModule
    from vap_agent.audio_to_tensor_module import AudioToTensor

    mic = MicrophoneStereoModule()
    a2t = AudioToTensor()
    tsend = TensorZMQModule(topic="data", port=5557)
    mic.subscribe(a2t)
    a2t.subscribe(tsend)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
