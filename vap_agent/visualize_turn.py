from argparse import ArgumentParser
from os.path import join, dirname, basename
from glob import glob
import torch
import matplotlib.pyplot as plt

import streamlit as st

from vap.audio import load_waveform, log_mel_spectrogram
from vap.utils import read_json
from vap.plot_utils import plot_stereo_mel_spec

parser = ArgumentParser()
parser.add_argument("--root", type=str, default="runs")
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


def get_sessions(root):
    audio_files = glob(join(root, "**/*.wav"), recursive=True)
    sessions_to_paths = {}
    for af in audio_files:
        session_path = dirname(af)
        session = basename(session_path)
        sessions_to_paths[session] = session_path

    return sessions_to_paths


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


def plot_vap(vap, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    else:
        fig = None
    ax.plot(vap["time"], vap["p_now"], color="r", alpha=0.6, label="P-now")
    ax.plot(
        vap["time"],
        vap["p_future"],
        linestyle="dashed",
        color="darkred",
        alpha=0.6,
        label="P-future",
    )
    ax.axhline(0.5, linewidth=1, linestyle="dashed", color="k")
    ax.set_xlim([0, vap["time"][-1]])
    ax.set_yticks([])
    ax.legend()
    return fig


def plot_bc(vap):
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    ax.plot(vap["time"], vap["p_bc_a"], alpha=0.6, color="b", label="P-BC-A")
    ax.plot(vap["time"], vap["p_bc_b"], alpha=0.6, color="orange", label="P-BC-B")
    ax.legend()
    ax.set_xlim([0, vap["time"][-1]])
    ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_waveform(y, sr=16_000, max_points=1000, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    else:
        fig = None

    hop_length = max(1, y.shape[-1] // max_points)
    y_env = y.unfold(dimension=-1, size=hop_length, step=hop_length)
    y_env = y_env.abs().max(dim=-1).values

    duration = y.shape[-1] / sr
    n_frames = y_env.shape[-1]
    s_per_frame = duration / n_frames
    x = torch.arange(0, duration, s_per_frame)
    x = x[:n_frames]
    ax.fill_between(x, -y_env[0], y_env[0], alpha=0.6, color="b")
    ax.fill_between(x, -y_env[1], y_env[1], alpha=0.6, color="orange")
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim([x[0], x[-1]])
    plt.tight_layout()
    return fig, ax


def plot_mel_spectrogram(
    y, sample_rate=16_000, hop_time=0.02, frame_time=0.05, n_mels=80, ax=None
):
    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    else:
        fig = None

    duration = y.shape[-1] / sample_rate
    xmin, xmax = 0, duration
    ymin, ymax = 0, 80

    hop_length = round(sample_rate * hop_time)
    frame_length = round(sample_rate * frame_time)
    spec = log_mel_spectrogram(
        y,
        n_mels=n_mels,
        n_fft=frame_length,
        hop_length=hop_length,
        sample_rate=sample_rate,
    )
    ax[0].imshow(
        spec[0],
        interpolation="none",
        aspect="auto",
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
    )
    ax[1].imshow(
        spec[1],
        interpolation="none",
        aspect="auto",
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
    )

    if fig is not None:
        plt.subplots_adjust(
            left=0.05, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0
        )
    return fig


def plot_full_global(waveform, vap, dialog):
    fig, ax = plt.subplots(4, 1, figsize=(16, 8), sharex=True)
    plot_mel_spectrogram(waveform, ax=[ax[0], ax[1]])
    plot_waveform(waveform, max_points=2000, ax=ax[2])
    plot_vap(vap, ax=ax[-1])
    for a in ax:
        a.set_yticks([])
    plt.subplots_adjust(
        left=0.01, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0
    )
    for turn in dialog:
        color = "b"
        if turn["speaker"] == "b":
            color = "orange"
        ax[-1].fill_betweenx(
            y=[0, 1], x1=turn["start"], x2=turn["end"], color=color, alpha=0.1
        )
    # plt.pause(0.1)

    return fig


def debug():
    args.dirpath = "runs/test/sample"
    waveform = load_audio(join(args.dirpath, "audio.wav"))
    vap = load_vap_log(join(args.dirpath, "turn_taking_log.txt"))
    dialog = read_json(join(args.dirpath, "turn_taking.json"))


def load_data(dirpath):
    vap = load_vap_log(join(dirpath, "turn_taking_log.txt"))
    dialog = read_json(join(dirpath, "turn_taking.json"))
    waveform = load_audio(join(dirpath, "audio.wav"))
    return waveform, vap, dialog


if __name__ == "__main__":
    sess2path = get_sessions(args.root)
    sorted_sessions = list(sess2path.keys())
    sorted_sessions.sort(reverse=True)
    sess = st.selectbox("Session", sorted_sessions)
    dirpath = sess2path[sess]
    waveform, vap, dialog = load_data(dirpath)
    fig_global = plot_full_global(waveform, vap, dialog)
    st.title(f"Session: {dirpath}")
    st.pyplot(fig_global)
    st.audio(open(join(dirpath, "audio.wav"), "rb").read(), format="audio/wav")
