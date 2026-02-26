# Data Collection Pipeline

## Docs

Documentation for the dvrk setup can be found in the [docs](docs/) directory.

## Scripts

Documentation for the scripts can be found [here](src/surpass_data_collection).

## A basic workflow (mainly for Grayson to quickly remember) is:

### Filter Data

```bash
python -m surpass_data_collection.scripts.sync_image_kinematics.filter_episodes --source_dir --out_dir
```

--source_dir = is the raw data  
--out_dir = where you want the filtered data to be saved  

Example:
```bash
python -m surpass_data_collection.scripts.sync_image_kinematics.filter_episodes ./Data ./FilteredData
```

### Slice Affordance

```bash
python -m surpass_data_collection.scripts.post_processing.slice_affordance \
 --cautery_dir ./FilteredData/Cholecystectomy/tissues \
 --post_process_dir ./Data/Cholecystectomy/post_annotation \
 --out_dir ./FilteredData/Cholecystectomy/tissues_sliced 
```

### Lerobot Conversion

```bash
python -m surpass_data_collection.scripts.lerobot_conversion.dvrk_zarr_to_lerobot \
    --data_path /home/grayson/surpass/data/FilteredData/Cholecystectomy/tissuees_sliced \
    --repo-id surpass/cholecystectomy
```