import zmq
import time
import msgpack
import msgpack_numpy as m

m.patch()


# Create ZMQ socket
port = 5558
context = zmq.Context()
INTERVAL = 0.1
topic = "new_speaker"

pair = False
if pair:
    socket = context.socket(zmq.PAIR)
    socket.connect(f"tcp://localhost:{port}")

    i = 0
    while True:
        msg = socket.recv()
        d = msgpack.unpackb(msg)
        print(d)
        time.sleep(INTERVAL)
        i += 1
        print(f"received {i}")
else:
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    i = 0
    while True:
        topic = socket.recv_string()
        # d = socket.recv_pyobj()
        d = socket.recv()
        print("received: ", type(d), d)
        time.sleep(INTERVAL)
        i += 1
