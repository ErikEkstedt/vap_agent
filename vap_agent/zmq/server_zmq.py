import zmq

"""
Server is paired to `audio_zmq.py`
"""

context = zmq.Context()


def simple_server():
    audio_in = context.socket(zmq.PAIR)
    audio_in.bind("tcp://*:5555")

    while True:
        message = audio_in.recv()
        print("Received message: ", message[:10])


if __name__ == "__main__":

    try:
        simple_server()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        context.destroy()
        print("Context destroyed")
