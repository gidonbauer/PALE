import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.integrate import simpson
import sys
from argparse import ArgumentParser
import json


def read_monitor_file(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=R'\s*\|\s*', header=0, skiprows=[1], index_col=False, engine='python')
    df = df.drop([df.columns[0], df.columns[-1]], axis=1)
    return df


def R(t, s):
  return 2*s['beta']*np.sqrt(s['alpha']*t)


def R_dot(t, s):
  return s['beta']*np.sqrt(s['alpha']/t)


def main():
    parser = ArgumentParser()
    parser.add_argument("monitor_file", type=str, help="Monitor file of the simulation.")
    parser.add_argument("setup_file", type=str, help="Setup file of the simulation in JSON format.")
    args = parser.parse_args()

    df = read_monitor_file(args.monitor_file)
    try:
        with open(args.setup_file, "r") as f:
            s = json.load(f)
    except IOError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    R_L1 = simpson(np.abs(df['r'] - R(df['t'], s)), df['t'])
    R_dot_L1 = simpson(np.abs(df['r_dot'] - R_dot(df['t'], s)), df['t'])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 5), layout="tight")

    # - Radius ---------------------------------------------
    ax[0, 0].plot(df['t'], df['r']*1e6, label="Simulation")

    ts = np.linspace(s['t0'], s['tend'], 1000)
    Rs = R(ts, s)
    ax[0, 0].plot(ts, Rs*1e6, linestyle='--', label="Analytical")

    ax[0, 0].set_xlabel('Time [s]')
    ax[0, 0].set_ylabel('Radius [µm]')
    ax[0, 0].legend()
    ax[0, 0].annotate(f'$L_1$-error = {R_L1:.8f}',
                      xy=(0.55, 0.1),
                      xycoords='axes fraction',
                      bbox=dict(facecolor='none', edgecolor='black'))
    # - Radius ---------------------------------------------

    # - Change of radius -----------------------------------
    ax[1, 0].plot(df['t'], df['r_dot']*1e6, label="Simulation")

    R_dots = R_dot(ts, s)
    ax[1, 0].plot(ts, R_dots*1e6, linestyle='--', label="Analytical")

    ax[1, 0].set_xlabel('Time [s]')
    ax[1, 0].set_ylabel('Change of radius [µm/s]')
    ax[1, 0].legend()
    ax[1, 0].annotate(f'$L_1$-error = {R_dot_L1:.8f}',
                      xy=(0.05, 0.1),
                      xycoords='axes fraction',
                      bbox=dict(facecolor='none', edgecolor='black'))
    # - Change of radius -----------------------------------

    # - Effective beta -------------------------------------
    ax[0, 1].plot(df['t'], df['beta_eff'], label="Simulation")
    ax[0, 1].plot([s['t0'], s['tend']],
                  [s['beta'], s['beta']],
                  linestyle='--', label="Analytical")

    ax[0, 1].set_xlabel('Time [s]')
    ax[0, 1].set_ylabel(R'Effective $\beta$ [-]')
    ax[0, 1].legend()
    # - Effective beta -------------------------------------

    plt.show()


if __name__ == "__main__":
    main()
