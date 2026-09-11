import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as rxr
from matplotlib.lines import Line2D

from processing_data.loading_impact_data import load_admin_boundaries


def plot_map(farm_path, label_farm, admin_path, output):

    #load basemap
    basemap = gpd.read_file(admin_path)

    #load farm data, masked true handels nodata pixels
    farm_data = rxr.open_rasterio(farm_path, masked=True)

    #show farm data only over ssd
    farm_ssd = farm_data.rio.clip(basemap.geometry, basemap.crs, drop=True, from_disk=True)

    #make nodata values transparant
    farm_ssd = farm_ssd.where(farm_ssd > 0)

    _, ax = plt.subplots(figsize=(12,8))

    basemap.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=1.5)


    calc_vmax = float(farm_ssd.quantile(0.95))
    if calc_vmax <= 0:
        vmax = float(farm_ssd.max())
    else:
        vmax = calc_vmax

    #plot heatmap
    farm_ssd[0].plot(
        ax=ax,
        cmap='YlOrRd',
        vmin=0,
        vmax = vmax,
        cbar_kwargs={"label": label_farm}
    )

    basemap.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)


    admin1, admin2 = load_admin_boundaries()

    #Nothern Bahr el Ghazal
    nbeg_state = admin1[admin1['adm1_name'].str.contains('Northern Bahr el Ghazal', case=False, na=False)]

    #River lol, Aweil
    rla_counties = admin2[admin2['adm2_name'].str.contains('Aweil|Lol', case=False, na=False)]


    #plot oultines ontop
    nbeg_state.plot(ax=ax, facecolor='none', edgecolor='cyan', linewidth=3)
    rla_counties.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=1.5)



    legend_elements = [
        Line2D([0], [0], color='cyan', lw=3, label='Northern Bahr el Ghazal State'),
        Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Aweil Counties'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    ax.set_axis_off()
    plt.title(f" {label_farm} Distribution in South Sudan", fontsize=16, pad=15)
    plt.tight_layout()

    plt.savefig(output)
    print(f"map saved to : {output}")


    plt.show()
