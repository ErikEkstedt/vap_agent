from argparse import ArgumentParser
from os.path import join, dirname, basename, exists
from glob import glob
import torch
import matplotlib.pyplot as plt

import streamlit as st

from vap.audio import load_waveform, log_mel_spectrogram
from vap.utils import read_json, read_txt
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


def load_data(dirpath):
    vap = load_vap_log(join(dirpath, "turn_taking_log.txt"))
    dialog = read_json(join(dirpath, "turn_taking.json"))
    waveform, _ = load_waveform(join(dirpath, "audio.wav"))

    asr = None
    asr_path = join(dirpath, "asr.txt")
    if exists(asr_path):
        asr = read_txt(asr_path)
    return waveform, vap, dialog, asr


def load_vap_log(path):
    """
    TIME P-NOW P-FUTURE P-BC-A P-BC-B
    """
    d = {
        "time": [],
        "p_now": [],
        "p_future": [],
        "p_bc_a": [],
        "p_bc_b": [],
        "H": [],
        "vad": [],
        "v1": [],
        "v2": [],
    }
    log = read_txt(path)
    for row in log[1:]:
        t, p_now, p_fut, p_bc_a, p_bc_b, H, v1, v2 = row.split()

        d["time"].append(float(t))
        d["p_now"].append(float(p_now))
        d["p_future"].append(float(p_fut))
        d["p_bc_a"].append(float(p_bc_a))
        d["p_bc_b"].append(float(p_bc_b))
        d["H"].append(float(H))
        d["v1"].append(float(v1))
        d["v2"].append(float(v2))

    for k, v in d.items():
        d[k] = torch.tensor(v)

    return d


#################################################################
# Plots
#################################################################
def plot_turn_shift(vap, dialog, ax=None, lw=2):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    else:
        fig = None

    for turn in dialog:
        color = "b"
        if turn["speaker"] == "b":
            color = "orange"
        ax.fill_betweenx(
            y=[0, 1], x1=turn["start"], x2=turn["end"], color=color, alpha=0.1
        )
    ax.plot(
        vap["time"], vap["p_now"], linewidth=lw, color="r", alpha=0.6, label="P-now"
    )
    ax.plot(
        vap["time"],
        vap["p_future"],
        linestyle="dashed",
        linewidth=lw,
        color="darkred",
        alpha=0.6,
        label="P-future",
    )
    diff = vap["p_future"] - vap["p_now"]
    ax.fill_between(
        vap["time"],
        vap["p_future"],
        vap["p_now"],
        where=diff >= 0,
        color="b",
        alpha=0.2,
        label="P-diff",
    )
    ax.fill_between(
        vap["time"],
        vap["p_future"],
        vap["p_now"],
        where=diff <= 0,
        color="orange",
        alpha=0.2,
        label="P-diff",
    )

    ax.axhline(0.4, linewidth=2, linestyle="dotted", color="g", alpha=0.3, zorder=1)
    ax.axhline(0.5, linewidth=2, linestyle="dotted", color="k", zorder=1)
    ax.axhline(0.6, linewidth=2, linestyle="dotted", color="g", alpha=0.3, zorder=1)
    ax.set_xlim([0, vap["time"][-1]])
    ax.set_yticks([])
    return fig


def plot_information(vap, ax, lw=2, alpha=0.8):
    ax.plot(
        vap["time"],
        vap["H"],
        color="green",
        linewidth=lw,
        alpha=alpha,
        label="H entropy",
    )
    ax.set_ylim([0, 8])


def plot_bc(vap):
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    ax.plot(vap["time"], vap["p_bc_a"], alpha=0.6, color="b", label="P-BC-A")
    ax.plot(vap["time"], vap["p_bc_b"], alpha=0.6, color="orange", label="P-BC-B")
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
    ax.fill_between(x, -y_env[0], y_env[0], alpha=0.6, color="b", label="A")
    ax.fill_between(x, -y_env[1], y_env[1], alpha=0.6, color="orange", label="B")
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


def plot_vad(vap, ax, scale=1):
    ax[0].plot(vap["time"], vap["v1"] * scale, color="w", label="VAD")
    ax[1].plot(vap["time"], vap["v2"] * scale, color="w", label="VAD")


def plot_full_global(waveform, vap, dialog):
    fig, axs = plt.subplots(5, 1, figsize=(16, 10), sharex=True)

    lw = 3
    plot_mel_spectrogram(waveform, ax=[axs[0], axs[1]])
    plot_vad(vap, scale=80, ax=[axs[0], axs[1]])
    plot_waveform(waveform, max_points=2000, ax=axs[2])
    plot_turn_shift(vap, dialog, ax=axs[3], lw=lw)
    plot_information(vap, ax=axs[-1], lw=lw)

    for a in axs[:-1]:
        a.set_yticks([])

    for a in axs:  # not mels
        a.legend(loc="upper left", fontsize=12)

    plt.subplots_adjust(
        left=0.01, bottom=None, right=0.99, top=0.99, wspace=0.01, hspace=0
    )
    return fig


def debug():
    args.dirpath = "runs/test/sample"
    waveform = load_audio(join(args.dirpath, "audio.wav"))
    vap = load_vap_log(join(args.dirpath, "turn_taking_log.txt"))
    dialog = read_json(join(args.dirpath, "turn_taking.json"))


if __name__ == "__main__":
    sess2path = get_sessions(args.root)
    sorted_sessions = list(sess2path.keys())
    sorted_sessions.sort(reverse=True)
    sess = st.selectbox("Session", sorted_sessions)
    dirpath = sess2path[sess]
    waveform, vap, dialog, asr = load_data(dirpath)
    fig_global = plot_full_global(waveform, vap, dialog)
    st.title(f"Session: {dirpath}")
    st.pyplot(fig_global)
    st.audio(open(join(dirpath, "audio.wav"), "rb").read(), format="audio/wav")
    if asr is not None:
        st.write(asr)
