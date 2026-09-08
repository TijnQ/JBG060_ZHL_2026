import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray as rxr


def plot_map(farm_path, label_farm, admin_path, output):

    #load basemap
    basemap = gpd.read_file(admin_path)

    #load farm data, masked true handels nodata pixels
    farm_data = rxr.open_rasterio(farm_path, masked=True)

    #show farm data only over ssd
    farm_ssd = farm_data.rio.clip(basemap.geometry, basemap.crs, drop=True)

    _, ax = plt.subplots(figsize=(12,8))

    basemap.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=1.5)


    calc_vmax = float(farm_ssd.quantile(0.95))
    if calc_vmax <= 0:
        vmax = float(farm_ssd.max())
    else:
        vmax = calc_vmax


    farm_ssd[0].plot(
        ax=ax,
        cmap='YlOrRd',
        vmin=0,
        vmax = vmax,
        cbar_kwargs={"label": label_farm}
    )

    basemap.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

    ax.set_axis_off()
    plt.title(f" {label_farm} Distribution in South Sudan", fontsize=16, pad=15)
    plt.tight_layout()

    plt.savefig(output)
    print(f"map saved to : {output}")


    plt.show()
