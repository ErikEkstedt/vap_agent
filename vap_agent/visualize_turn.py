from argparse import ArgumentParser
from os.path import join
import torch
import matplotlib.pyplot as plt

import streamlit as st

from vap.audio import load_waveform, log_mel_spectrogram
from vap.plot_utils import plot_stereo_mel_spec

parser = ArgumentParser()
parser.add_argument(
    "--dirpath",
    type=str,
    # default="runs/test/sample",
)
try:
    args = parser.parse_args()
except SystemExit as e:
    # This exception will be raised if --help or invalid command line arguments
    # are used. Currently streamlit prevents the program from exiting normally
    # so we have to do a hard exit.
    import os

    os._exit(0)


def read_txt(path, encoding="utf-8"):
    data = []
    with open(path, "r", encoding=encoding) as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def load_vap_log(path):
    """
    TIME P-NOW P-FUTURE P-BC-A P-BC-B
    """
    d = {"time": [], "p_now": [], "p_future": [], "p_bc_a": [], "p_bc_b": []}
    log = read_txt(path)
    for row in log[1:]:
        t, p_now, p_fut, p_bc_a, p_bc_b = row.split()
        d["time"].append(float(t))
        d["p_now"].append(float(p_now))
        d["p_future"].append(float(p_fut))
        d["p_bc_a"].append(float(p_bc_a))
        d["p_bc_b"].append(float(p_bc_b))

    for k, v in d.items():
        d[k] = torch.tensor(v)

    return d


def load_audio(path):
    y, sr = load_waveform(path)
    return y


def plot_vap(vap):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(vap["time"], vap["p_now"], color="r", alpha=0.6, label="P-now")
    ax[0].plot(
        vap["time"],
        vap["p_future"],
        linestyle="dashed",
        color="darkred",
        alpha=0.6,
        label="P-future",
    )
    ax[0].axhline(0.5, linewidth=1, linestyle="dashed", color="k")
    ax[1].plot(vap["time"], vap["p_bc_a"], alpha=0.6, color="b", label="P-BC-A")
    ax[1].plot(vap["time"], vap["p_bc_b"], alpha=0.6, color="orange", label="P-BC-B")
    for a in ax:
        a.legend()
    plt.tight_layout()
    return fig


if __name__ == "__main__":

    if "vap" not in st.session_state:
        st.session_state.vap = load_vap_log(join(args.dirpath, "turn_taking_log.txt"))

    if "waveform" not in st.session_state:
        st.session_state.waveform = load_audio(join(args.dirpath, "audio.wav"))

    if "fig_global_audio" not in st.session_state:
        fig, ax = plt.subplots(2, 1, figsize=(12, 4))
        plot_stereo_mel_spec(st.session_state.waveform, ax=ax, plot=False)
        st.session_state.fig_global_audio = fig

    if "fig_global_vap" not in st.session_state:
        st.session_state.fig_global_vap = plot_vap(st.session_state.vap)

    st.title(f"Session: {args.dirpath}")

    st.pyplot(st.session_state.fig_global_audio)
    st.pyplot(st.session_state.fig_global_vap)
