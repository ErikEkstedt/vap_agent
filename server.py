from argparse import ArgumentParser
import matplotlib.pyplot as plt
import threading
import torch
import zmq
import time


NORM_FACTOR = 1 / (2 ** 15)
ctx1 = zmq.Context(1)
ctx2 = zmq.Context(2)
print("Connecting to ZeroMQ server")


class AudioMixer:
    def __init__(
        self, port1, port2, frame_time=0.02, buffer_time=1, sample_rate=16_000
    ):
        self.port1 = port1
        self.port2 = port2

        self.sample_rate = sample_rate
        self.buffer_time = buffer_time
        self.chunk_size = int(frame_time * self.sample_rate)
        self.n_samples = int(buffer_time * self.sample_rate)
        self.x = torch.zeros((1, 2, self.n_samples))

        self.bind()

        self._active = False

    def bind(self):
        print("Port 1: ", self.port1)
        self.a_socket = ctx1.socket(zmq.PAIR)
        self.a_socket.bind(f"tcp://*:{self.port1}")

        print("Port 2: ", self.port2)
        self.b_socket = ctx2.socket(zmq.PAIR)
        self.b_socket.bind(f"tcp://*:{self.port2}")

    def update_tensor(self, a, b):
        xa = torch.frombuffer(a, dtype=torch.int16).float() * NORM_FACTOR
        xb = torch.frombuffer(b, dtype=torch.int16).float() * NORM_FACTOR

        # Move older samples back
        self.x[..., : -self.chunk_size] = self.x[..., self.chunk_size :]

        # Update new values
        self.x[0, 0, -self.chunk_size :] = xa
        self.x[0, 1, -self.chunk_size :] = xb

    def run(self):
        fig, [ax1, ax2] = plt.subplots(2, 1)
        (line1,) = ax1.plot(self.x[0, 0, ::10])
        (line2,) = ax2.plot(self.x[0, 1, ::10])
        ax1.set_ylim([-1, 1])
        ax2.set_ylim([-1, 1])
        plt.show(block=False)

        i = 0
        try:
            while True:
                a = self.a_socket.recv()
                b = self.b_socket.recv()
                self.update_tensor(a, b)

                i += 1
                if i % 10 == 0:
                    line1.set_ydata(self.x[0, 0, ::10])
                    line2.set_ydata(self.x[0, 1, ::10])
                    fig.canvas.draw()
                    fig.canvas.flush_events()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

        self._active = False
        print("MAIN THREAD ENDED")
        self.shutdown()

    # def _print_thread(self):
    #     while self._active:
    #         time.sleep(0.2)
    #         a = round(self.x[0, 0].abs().max().item(), 4)
    #         b = round(self.x[0, 1].abs().max().item(), 4)
    #         print(a, " <----> ", b)
    #     print("PRINT THREAD ENDED")

    # def prepare_run(self):
    #     self._active = True
    #     self.t = threading.Thread(target=self._print_thread)
    #     self.t.start()

    def shutdown(self):
        self.a_socket.close()
        print("a_socket CLOSE")
        self.b_socket.close()
        print("b_socket CLOSE")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-p1", "--port1", default=5555)
    parser.add_argument("-p2", "--port2", default=5556)
    args = parser.parse_args()
    mix = AudioMixer(port1=args.port1, port2=args.port2)

    # mix.prepare_run()
    mix.run()
