import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from processing_data.loading_impact_data import load_admin_boundaries, load_ipc_data


def plot_map_ipc(output):

    admin1, admin2 = load_admin_boundaries()
    ipc_data = load_ipc_data()


    latest_period = ipc_data.iloc[-1]
    start_date = latest_period['Start Date'].date()
    end_date = latest_period['End Date'].date()


    # Drop the date columns to create a clean County -> Population dataframe
    county_ipc = latest_period.drop(['Start Date', 'End Date']).reset_index()
    county_ipc.columns = ['County', 'Phase 3+ Pop']

    #convert txt to numbers
    county_ipc['Phase 3+ Pop'] = pd.to_numeric(county_ipc['Phase 3+ Pop'], errors='coerce')

    #merge ipc with county geodataframe
    merged_map = admin2.merge(county_ipc, left_on='adm2_name', right_on='County', how='left')

    #stakeholders places
    nbeg_state = admin1[admin1['adm1_name'].str.contains('Northern Bahr', case=False, na=False)]
    rla_counties = admin2[admin2['adm2_name'].str.contains('Aweil|Lol', case=False, na=False)]

    _, ax = plt.subplots(figsize=(12, 8))

    merged_map.plot(
    column='Phase 3+ Pop',
    ax=ax,
    cmap='OrRd',
    edgecolor='black',
    linewidth=0.5,
    legend=True,
    missing_kwds={'color': 'lightgrey', 'label': 'No Data'},
    legend_kwds={'label': 'Population in IPC Phase 3+ (Crisis/Emergency/Famine)'}
    )


    #plot oultines ontop
    nbeg_state.plot(ax=ax, facecolor='none', edgecolor='cyan', linewidth=3)
    rla_counties.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1.5)

    legend_elements = [
    Line2D([0], [0], color='cyan', lw=3, label='Northern Bahr el Ghazal State'),
    Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Aweil Counties')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    ax.set_axis_off()
    plt.title(f"Acute Food Insecurity Hotspots in South Sudan\nPeriod: {start_date} to {end_date}", fontsize=16, pad=15)
    plt.tight_layout()

    plt.savefig(output)
    print(f"map saved to : {output}")

    plt.show()