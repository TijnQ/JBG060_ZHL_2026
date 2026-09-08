from pathlib import Path

from plot_map import plot_map

root_dir = Path(__file__).parent.parent

#basemap
admin_path = root_dir / 'raw_data/Administrative boundaries/ssd_admin1.geojson'





#cattle data
cattle_path = root_dir / 'raw_data/farmland/geonode__cattle_gha.tif'
cattle_output = root_dir / 'output/cattle_heatmap.png'

#crop data
crop_path = root_dir / 'raw_data/farmland/asap_mask_crop_v04.tif'
crop_output = root_dir / 'output/crop_heatmap.png'

plot_map(cattle_path, 'Cattle', admin_path, cattle_output)
plot_map(crop_path, 'Crop', admin_path, crop_output)