# https://gist.github.com/telegraphic/2709b7e6edc3a0c39ed9b75452da205e
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import zmq

REFRESH_TIME_MS = 200


def recv_zmq(port: int = 5557, topic: str = "data"):
    """Receive data over ZMQ PubSub socket
    Args:
        topic: topic subscribed to
    Returns numpy array data
    """

    def create_zmq_socket(port: int = 5557, topic: str = "data") -> zmq.Socket:
        """Create a ZMQ SUBSCRIBE socket"""
        context = zmq.Context()
        zmq_socket = context.socket(zmq.SUB)
        zmq_socket.connect(f"tcp://localhost:{port}")
        zmq_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        return zmq_socket

    # Note - context managing socket as it can't be shared between threads
    # This makes sure the socket is opened and closed by whatever thread Dash gives it
    with create_zmq_socket(port=port, topic=topic) as socket:
        # For pub/sub patterns, you must first send the topic as a string before
        # you send a pyobj.
        topic = socket.recv_string()
        data = socket.recv_pyobj()
    return data


external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    children=[
        html.H1(children="Agent"),
        html.Div(
            children="""
        Dash: A web application framework for Python.
    """
        ),
        dcc.Graph(
            id="example-graph",
            figure={
                "data": [
                    {
                        "x": [1, 2, 3, 4, 5],
                        "y": [4, 1, 2, 1, 2],
                        "type": "scatter",
                        "name": "SF",
                    },
                ],
                "layout": {"title": "Dash Data Visualization"},
            },
        ),
        dcc.Interval(
            id="interval-component",
            interval=REFRESH_TIME_MS,
            n_intervals=0,  # in milliseconds
        ),
    ]
)

# The updating is done with this callback
@app.callback(
    Output("example-graph", "figure"), [Input("interval-component", "n_intervals")]
)
def update(n):
    d = recv_zmq(port=5557, topic="data")
    return {
        "data": [
            {"x": d["x"], "y": d["y"], "type": "scatter", "name": "SF"},
        ]
    }


if __name__ == "__main__":
    app.run_server(debug=True)
