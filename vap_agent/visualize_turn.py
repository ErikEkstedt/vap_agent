from argparse import ArgumentParser
from os.path import join, dirname, basename, exists
from glob import glob
import torch
import matplotlib.pyplot as plt

import streamlit as st

import vap.functional as VF
from vap.audio import load_waveform, log_mel_spectrogram
from vap.utils import read_json, read_txt
from vap.plot_utils import plot_stereo_mel_spec
from vap_turn_taking.vap_new import VAP

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

    d["vad"] = torch.stack([d["v1"], d["v1"]], dim=-1)

    # vad = (vad > 0.5).float()
    # zero_pad = torch.zeros((100, 2))
    # vad = torch.cat((vad, zero_pad), dim=0)

    # vapper = VAP()
    # d["labels"] = vapper.extract_labels(vad)
    # print("d['labels']: ", tuple(d["labels"].shape))
    # print(d["labels"])
    return d


#################################################################
# Plots
#################################################################
def plot_turn_shift(vap, dialog, ax=None, lw=2):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    else:
        fig = None

    # Color background based on turns

    speaker_set = {"a": False, "b": False}
    for turn in dialog:

        current_speaker = turn["speaker"]
        color = "b"
        if current_speaker == "b":
            color = "orange"

        # find times in this turn
        after_start = turn["start"] <= vap["time"]
        before_end = vap["time"] <= turn["end"]
        in_idx = torch.where(torch.logical_and(after_start, before_end))

        if len(in_idx[0]) > 0:
            t = vap["time"][in_idx]
            bc_color = "darkgreen"
            if current_speaker == "b":
                v = vap["p_bc_a"][in_idx]
            else:
                v = vap["p_bc_b"][in_idx]
            ax.fill_between(t, v, 0, color=bc_color, alpha=0.3)

        if not speaker_set[turn["speaker"]]:
            ax.fill_betweenx(
                y=[0, 1],
                x1=turn["start"],
                x2=turn["end"],
                color=color,
                alpha=0.1,
                # hatch="+",
                label="VAD " + turn["speaker"].upper(),
            )
            speaker_set[turn["speaker"]] = True
        else:
            ax.fill_betweenx(
                y=[0, 1],
                x1=turn["start"],
                x2=turn["end"],
                color=color,
                alpha=0.1,
                # hatch="+",
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
        alpha=0.1,
        # label="A future",
    )
    ax.fill_between(
        vap["time"],
        vap["p_future"],
        vap["p_now"],
        where=diff <= 0,
        color="orange",
        alpha=0.1,
        # label="B future",
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


def plot_prosody(waveform, vap, dialog, ax, sample_rate=16000, hop_time=0.01):
    f0 = VF.pitch_praat(
        waveform.cpu(), sample_rate=sample_rate, hop_time=hop_time, f0_max=400
    )
    f0_times = torch.arange(f0.shape[-1]).float() * hop_time
    f0[f0 == 0] = torch.nan
    # ax.plot(
    #     f0_times,
    #     f0[1].log(),
    #     "o",
    #     markersize=4,
    #     color="orange",
    #     alpha=0.8,
    #     label="B F0",
    # )

    #######################################
    # F0
    #######################################
    # Plot only F0 for the current speaker
    # F0-praat: can't handle overbleed
    speaker_set = {"a": False, "b": False}  # only set legend once
    for turn in dialog:
        current_speaker = turn["speaker"]

        # find times in this turn
        after_start = turn["start"] <= f0_times
        before_end = f0_times <= turn["end"]
        in_idx = torch.where(torch.logical_and(after_start, before_end))

        # If there are frames in this turn we plot them
        if len(in_idx[0]) > 0:
            t = f0_times[in_idx]
            color = "b"
            if current_speaker == "a":
                f = f0[0][in_idx]
            else:
                f = f0[1][in_idx]
                color = "orange"

            if not speaker_set[current_speaker]:
                ax.plot(
                    t,
                    f.log(),
                    "o",
                    markersize=4,
                    color=color,
                    alpha=0.8,
                    label=current_speaker.upper() + " F0",
                )
                speaker_set[current_speaker] = True
            else:
                ax.plot(
                    t,
                    f.log(),
                    "o",
                    markersize=4,
                    color=color,
                    alpha=0.8,
                )

    #######################################
    # Plot F0 envelope?
    # env = f0.unfold(dimension=-1, step=1, size=10)
    # env = env.max(dim=-1).values
    # t_env = times.unfold(dimension=-1, step=1, size=10).max(dim=-1).values
    # ax.plot(t_env, env[0], color="b", alpha=0.5)
    # ax.plot(t_env, env[1], color="orange", alpha=0.5)

    #######################################
    # Intensity
    #######################################
    int_hop_time = 0.1
    ints = VF.intensity_praat(waveform, sample_rate=sample_rate, hop_time=int_hop_time)
    with torch.no_grad():
        k = 15
        c = torch.nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            stride=1,
            bias=False,
            padding="same",
            padding_mode="replicate",
        )
        c.weight.data.fill_(1 / k)
        ints = c(ints.unsqueeze(1))[:, 0]
    ints_times = torch.arange(ints.shape[-1]).float() * int_hop_time

    aa = ax.twinx()
    # aa = ax
    aa.plot(
        ints_times,
        ints[0].log(),
        linestyle="dashed",
        linewidth=3,
        color="b",
        alpha=0.2,
        label="A intensity",
    )
    aa.plot(
        ints_times,
        ints[1].log(),
        linestyle="dashed",
        linewidth=3,
        color="orange",
        alpha=0.4,
        label="B intensity",
    )
    aa.legend(loc="upper left")
    aa.set_yticks([])
    return ax


def plot_full_global(waveform, vap, dialog):
    fig, axs = plt.subplots(6, 1, figsize=(16, 10), sharex=True)

    lw = 3
    plot_mel_spectrogram(waveform, ax=[axs[0], axs[1]])
    plot_vad(vap, scale=80, ax=[axs[0], axs[1]])
    plot_waveform(waveform, max_points=2000, ax=axs[2])
    plot_turn_shift(vap, dialog, ax=axs[3], lw=lw)
    plot_prosody(waveform, vap, dialog, ax=axs[4])
    plot_information(vap, ax=axs[-1], lw=lw)

    for a in axs[:-1]:
        a.set_yticks([])

    for a in axs:  # not mels
        # a.legend(loc="upper left", fontsize=12)
        # a.legend(loc="lower left", fontsize=12)
        a.legend(loc="upper right", fontsize=12)

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
