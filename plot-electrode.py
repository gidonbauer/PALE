import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from argparse import ArgumentParser


def read_monitor_file(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=R'\s*\|\s*', header=0, skiprows=[1], index_col=False, engine='python')
    df = df.drop([df.columns[0], df.columns[-1]], axis=1)
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument("filename", type=str, help="Input file")
    args = parser.parse_args()

    df = read_monitor_file(args.filename)
    df["dsdt"] = np.gradient(df["sum(s)"], df["t"])
    # df["exp. dsdt"] = np.gradient(df["exp. sum(s)"], df["t"])

    _, ax = plt.subplots(nrows=2, figsize=(10, 5), layout='tight')

    ax[0].plot(df["t"], df["sum(s)"], label="Simulation")
    ax[0].plot(df["t"], df["exp. sum(s)"], linestyle="--", label="Expected")

    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Total species")
    ax[0].legend()

    step = int(1e-1/df["dt"].mean())
    t = df["t"].to_numpy()[::step]
    s = df["sum(s)"].to_numpy()[::step]
    dsdt = np.gradient(s, t)

    ax[1].plot(t, dsdt, label="Simulation")
    dsdt_true = np.sqrt(2**2 - 1.5**2)*1e-2*1.0
    ax[1].plot([df["t"].min(), df["t"].max()], [dsdt_true]*2, linestyle="--", label="Expected")
    # ax[1].plot(df["t"], df["dsdt"], label="Simulation")
    # ax[1].plot(df["t"], df["exp. dsdt"], linestyle="--", label="Expected")

    ax[1].set_xlabel("Time")
    ax[1].set_ylabel(R"$\partial s_T/\partial t$")
    ax[1].legend()
    ax[1].set_ylim([dsdt_true*0.99, dsdt_true*1.01])

    plt.show()

    _, ax_abs = plt.subplots(figsize=(10, 5), layout='tight')
    ax_rel = ax_abs.twinx()

    t = df["t"].to_numpy()
    s = df["sum(s)"].to_numpy()
    se = df["exp. sum(s)"].to_numpy()

    abs_line, = ax_abs.plot(t, np.abs(s - se), color="C0", label="Abs. error")
    rel_line, = ax_rel.plot(t[1:], np.abs(s[1:] - se[1:])/se[1:], color="C1", label="Rel. error")

    ax_abs.set_xlabel("Time")
    ax_abs.set_ylabel("Absolute error")
    ax_abs.tick_params(axis="y")
    ax_rel.set_ylabel("Relative error")
    ax_rel.tick_params(axis="y")
    ax_abs.legend(handles=[abs_line, rel_line], loc="upper center")

    plt.show()


if __name__ == "__main__":
    main()
