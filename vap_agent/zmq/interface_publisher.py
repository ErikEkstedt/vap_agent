import retico_core
import zmq

# https://gist.github.com/telegraphic/2709b7e6edc3a0c39ed9b75452da205e


class InterfaceZMQ(retico_core.AbstractConsumingModule):

    """A ZeroMQ Writer Module
    Note: If you are using this to pass IU payloads to PSI, make sure you're passing JSON-formatable stuff (i.e., dicts not tuples)
    Attributes:
    topic (str): topic/scope that this writes to
    """

    @staticmethod
    def name():
        return "ZeroMQ Writer Module"

    @staticmethod
    def description():
        return "A Module providing writing onto a ZeroMQ bus"

    @staticmethod
    def input_ius():
        return [retico_core.IncrementalUnit]

    def __init__(self, topic, **kwargs):
        """Initializes the ZeroMQReader.
        Args: topic(str): the topic/scope where the information will be read.
        """
        super().__init__(**kwargs)
        self.topic = topic.encode()

    def process_update(self, update_message):
        """
        This assumes that the message is json formatted, then packages it as payload into an IU
        """
        for um in update_message:
            self.queue.append(um)

        return None

    def run_writer(self):
        while True:
            if len(self.queue) == 0:
                time.sleep(0.1)
                continue
            input_iu, ut = self.queue.popleft()
            payload = {}
            payload["originatingTime"] = datetime.datetime.now().isoformat()

            # print(input_iu.payload)
            # if isinstance(input_iu, ImageIU) or isinstance(input_iu, DetectedObjectsIU)  or isinstance(input_iu, ObjectFeaturesIU):
            # payload['message'] = json.dumps(input_iu.get_json())
            # else:
            payload["message"] = json.dumps(input_iu.payload)
            payload["update_type"] = str(ut)

            self.writer.send_multipart(
                [self.topic, json.dumps(payload).encode("utf-8")]
            )

    def setup(self):
        self.socket = zmq.Context().socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")
        t = threading.Thread(target=self.run_writer)
        t.start()
