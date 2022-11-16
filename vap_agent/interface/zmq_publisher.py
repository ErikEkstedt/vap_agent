import zmq
import time
import msgpack
import msgpack_numpy as m
import numpy as np
from tqdm import tqdm

m.patch()

port = 5557
context = zmq.Context()
INTERVAL = 0.1

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
    topic = "data"

    i = 0
    pbar = tqdm(desc="publish data")
    while True:
        d = {"x": np.arange(512), "y": np.random.randint(10, size=512)}
        socket.send_string(topic, zmq.SNDMORE)
        socket.send_pyobj(d)
        time.sleep(INTERVAL)
        i += 1
        # print(f"sent {i}")
        pbar.update()
