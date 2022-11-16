import zmq
import pyaudio
import time
from argparse import ArgumentParser

"""
A simple audio chunk sender
"""

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
FRAME_TIME = 0.02
CHUNK_SIZE = int(FRAME_TIME * SAMPLE_RATE)


def callback(in_data, frame_count, time_info, status):
    """The callback function that gets called by pyaudio.
    Args:
        in_data (bytes[]): The raw audio that is coming in from the
            microphone
        frame_count (int): The number of frames that are stored in in_data
    """
    audio_socket.send(in_data)
    return (in_data, pyaudio.paContinue)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5555)
    parser.add_argument("-ip", "--ip", default="localhost")
    args = parser.parse_args()
    context = zmq.Context()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(SAMPLE_WIDTH),
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        output=False,
        stream_callback=callback,
        frames_per_buffer=CHUNK_SIZE,
        start=False,
    )

    print("Connecting to ZeroMQ server")
    print("IP: ", args.ip)
    print("Port: ", args.port)
    audio_socket = context.socket(zmq.PAIR)
    audio_socket.connect(f"tcp://{args.ip}:{args.port}")

    stream.start_stream()
    print("Start Stram")

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    finally:
        stream.stop_stream()
        print("Stream stopped")
        stream.close()
        print("Stream closed")
        # End zmq
        audio_socket.close()
        print("Socket closed")
        # context.destroy()
