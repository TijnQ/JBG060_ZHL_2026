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

    # --- check how often NBeG counties end up grey (no data) ---
    nbeg_map = merged_map[
        merged_map['adm1_name'].str.contains('Northern Bahr el Ghazal', case=False, na=False)
    ][['adm2_name', 'Phase 3+ Pop']].copy()
    nbeg_map['is_grey'] = nbeg_map['Phase 3+ Pop'].isna()
    print(nbeg_map)
    print(f"NBeG counties grey in latest period: {nbeg_map['is_grey'].sum()} of {len(nbeg_map)}")

    # why? does the missing county exist in the IPC data under a slightly different name?
    ipc_counties = set(county_ipc['County'].dropna())
    for county in nbeg_map.loc[nbeg_map['is_grey'], 'adm2_name']:
        candidates = [c for c in ipc_counties if str(county).split()[0].lower() in str(c).lower()]
        print(f"{county}: {'possible name match -> ' + str(candidates) if candidates else 'not in IPC data at all'}")

        # how many NBeG counties are missing in every assessment round
    nbeg_names = set(nbeg_map['adm2_name'].str.lower())
    nbeg_cols = [c for c in ipc_data.columns
                 if str(c).lower() in nbeg_names and c not in ['Start Date', 'End Date']]
    missing_per_round = ipc_data[nbeg_cols].apply(pd.to_numeric, errors='coerce').isna().sum(axis=1)
    print(pd.DataFrame({
        'Start Date': ipc_data['Start Date'],
        'Missing NBeG counties': missing_per_round,
    }))

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