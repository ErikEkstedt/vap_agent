import zmq
import time
import msgpack
import msgpack_numpy as m
import numpy as np
from tqdm import tqdm
import json

m.patch()

topic = "vap"
port = 5557
context = zmq.Context()
INTERVAL = 0.02


pair = False
if pair:
    socket = context.socket(zmq.PAIR)
    socket.bind(f"tcp://*:{port}")

    i = 0
    while True:
        messagedata = {"x": np.arange(1024), "y": np.random.randint(5, size=1024)}
        msg = msgpack.packb(messagedata)
        socket.send(msg)
        time.sleep(INTERVAL)
        i += 1
        print(f"sent {i}")
else:
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")

    i = 0
    pbar = tqdm(desc="publish data")
    while True:
        # d = {"x": np.arange(5), "y": np.random.randint(20, size=5)}
        d = {
            "x": [1, 2, 3, 4],
            "y": [10, 11, 12, 13],
            "p": np.random.rand(
                1,
            ).item(),
        }
        # d = b"hello world"
        # d = bytes([1, 2, 3, 4])
        # d = msgpack.packb(d)
        # d = bytes(d)
        d = json.dumps(d).encode()

        socket.send_string(topic, zmq.SNDMORE)
        # socket.send_pyobj(d)
        socket.send(d)

        time.sleep(INTERVAL)
        i += 1
        # print(f"sent {i}")
        pbar.update()
