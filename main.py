import retico_core

from microphone_zmq_module import MicrophoneZMQModule
from audio_collector_module import AudioCollector
from vap_module import VapModule


if __name__ == "__main__":
    mic = MicrophoneZMQModule(speaker=0, port=5555)
    mic2 = MicrophoneZMQModule(speaker=1, port=5556)
    coll = AudioCollector()
    vapper = VapModule()

    mic.subscribe(coll)
    mic2.subscribe(coll)
    coll.subscribe(vapper)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)
