import retico_core
from argparse import ArgumentParser

from microphone_zmq_module import MicrophoneZMQModule
from microphone_stereo_module import (
    MicrophoneStereoModule,
    AudioToTensor,
    audioTensorCallback,
)
from audio_collector_module import AudioCollector
from vap_module import VapModule

from vap.audio import log_mel_spectrogram


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--port1", default=5555)
    parser.add_argument("--port2", default=5556)
    parser.add_argument("--refresh_time", type=float, default=0.5)
    parser.add_argument("--buffer_time", type=float, default=2)
    parser.add_argument("--zmq", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main_zmq(args):
    print("ZMQ VAP Main")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    mic = MicrophoneZMQModule(speaker=0, port=args.port1)
    mic2 = MicrophoneZMQModule(speaker=1, port=args.port2)
    coll = AudioCollector()
    vapper = VapModule(refresh_time=args.refresh_time)

    mic.subscribe(coll)
    mic2.subscribe(coll)
    coll.subscribe(vapper)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)


def main(args):
    print("VAP Main")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    mic = MicrophoneStereoModule()
    vapper = VapModule(buffer_time=args.buffer_time, refresh_time=args.refresh_time)
    a2t = AudioToTensor(buffer_time=args.buffer_time, device=vapper.device)
    # clb = retico_core.debug.CallbackModule(audioTensorCallback)

    mic.subscribe(a2t)
    a2t.subscribe(vapper)
    # a2t.subscribe(clb)

    retico_core.network.run(mic)
    input()
    retico_core.network.stop(mic)


def main_debug(args):
    import torch
    import time
    import matplotlib.pyplot as plt

    print("VAP Main")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    mic = MicrophoneStereoModule()
    vapper = VapModule(buffer_time=args.buffer_time, refresh_time=args.refresh_time)
    a2t = AudioToTensor(buffer_time=args.buffer_time, device=vapper.device)
    mic.subscribe(a2t)
    a2t.subscribe(vapper)

    ############################################################
    plot = False
    if plot:
        fig, ax = plt.subplots(2, 1)
        z = torch.zeros((1, 2, int(16_000 * args.buffer_time)))
        l1 = ax[0].imshow(
            log_mel_spectrogram(z[0, 0]),
            aspect="auto",
            interpolation="none",
            origin="lower",
        )
        l2 = ax[1].imshow(
            log_mel_spectrogram(z[0, 1]),
            aspect="auto",
            interpolation="none",
            origin="lower",
        )
        ax[0].set_xticks([])
        ax[1].set_xticks([])

        l1.set_clim(vmin=-1.5, vmax=1.5)
        l2.set_clim(vmin=-1.5, vmax=1.5)
        print("PLOT")
        plt.pause(0.5)
    ############################################################

    retico_core.network.run(mic)
    # input()
    while True:
        try:
            time.sleep(0.2)
            if plot:
                # l1.set_ydata(a2t.tensor[0, 0, ::10])
                # l2.set_ydata(a2t.tensor[0, 1, ::10])
                l1.set_data(log_mel_spectrogram(a2t.tensor[0, 0].cpu()))
                l2.set_data(log_mel_spectrogram(a2t.tensor[0, 1].cpu()))
                fig.canvas.draw()
                fig.canvas.flush_events()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break
    retico_core.network.stop(mic)


if __name__ == "__main__":
    args = get_args()

    if args.zmq:
        main_zmq(args)
    elif args.debug:
        main_debug(args)
    else:
        main(args)
