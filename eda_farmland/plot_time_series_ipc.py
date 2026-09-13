import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from processing_data.loading_impact_data import load_admin_boundaries, load_ipc_data


def plot_time_series_ipc(output):

    _, admin2 = load_admin_boundaries()
    ipc_data = load_ipc_data()

    # get list of all counties (all columns except the date columns)
    county_cols = [col for col in ipc_data.columns if col not in ['Start Date', 'End Date']]

    # make data numeric
    for col in county_cols:
        ipc_data[col] = pd.to_numeric(ipc_data[col], errors='coerce')

    # select all counties of Northern Bahr el Ghazal from the admin2 boundaries
    # (instead of only Aweil, so the line represents the whole state)
    nbeg_counties = admin2.loc[
        admin2['adm1_name'].str.contains('Northern Bahr el Ghazal', case=False, na=False),
        'adm2_name'
    ].str.lower()
    valid_nbeg_cols = [col for col in county_cols if col.lower() in set(nbeg_counties)]

    print(f"Matched NBeG columns for the chart: {valid_nbeg_cols}")
    if not valid_nbeg_cols:
        print("No matches found - check the admin2 names:", admin2['adm2_name'].unique())
        return

    # sums over counties = total population in Phase 3+ for the state / the country
    ipc_data['South Sudan Total'] = ipc_data[county_cols].sum(axis=1)
    ipc_data['NBeG Total'] = ipc_data[valid_nbeg_cols].sum(axis=1)

    # sanity check: expect one row per assessment round (5 rounds, 2022-2025)
    print(ipc_data[['Start Date', 'End Date', 'NBeG Total', 'South Sudan Total']])

    # two panels with a shared x-axis: state and national scales differ too much for one axis
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    ax1.plot(ipc_data['Start Date'], ipc_data['NBeG Total'],
             label='Northern Bahr el Ghazal (Total Phase 3+)',
             color='red', marker='s', linewidth=2.5)
    ax1.set_ylabel("Population in Phase 3+", fontsize=12)

    ax2.plot(ipc_data['Start Date'], ipc_data['South Sudan Total'],
             label='South Sudan (Total Phase 3+)',
             color='grey', marker='o', linewidth=2, linestyle='--')
    ax2.set_ylabel("Population in Phase 3+", fontsize=12)
    ax2.set_xlabel("Assessment Start Date", fontsize=12)

    ax1.set_title("IPC Phase 3+: Northern Bahr el Ghazal vs. National trend", fontsize=16, pad=15)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    for ax in [ax1, ax2]:
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend(loc='best', fontsize=11, frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output)
    print(f"Time-series chart saved to: {output}")
    plt.show()