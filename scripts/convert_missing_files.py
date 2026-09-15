import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import data, movie_processing

#cycle through the null files 

from pathlib import Path
import json
import numpy as np
import pandas as pd
import tqdm 


def main():
    alldata = data.load_all_excels(drive='/Volumes/Data')

    alldata = alldata[alldata['discard2']!=1]

    dfdata = data.generate_dataset(alldata)

    dfdata = dfdata[dfdata['TIFF_file'].isnull()]

    pbar = tqdm.tqdm(total=len(dfdata), desc='Processing missing files', unit='file')
    zarr_list = []
    zarr_list_file = Path(__file__).with_name('processed_zarr_files.txt')
    zarr_list_file.write_text('')
    for j, el in dfdata.iterrows():
        # if j>39:
        #     continue
        pbar.update(1)
        #print(f'Processing {j+1}/{len(dfdata)}: {el["Folder"]}')
        base_folder = Path(el['Folder'])
        processed_folder = base_folder / 'processedMovies'

        ooffile = processed_folder / 'out_of_focus_labels.csv'
        jsonfile = processed_folder / 'processing_parameters.json'
        rawfile = base_folder / 'Image_001_001.raw'

        output_path = processed_folder / '1-jumpCorrected.ome.zarr'
        # output_path = processed_folder / '1-jumpCorrected.tif'
        # output_path = None

        with open(jsonfile, 'r') as f:
            params = json.load(f)

        oof = pd.read_csv(ooffile)
        width, height, raw_n_frames = movie_processing.getImgInfo(base_folder, filename=rawfile.name, n_channels=params['nChannels'])
        processed_n_frames = int(params['last_frame']) - int(params['first_frame']) + 1
        frames_to_process = processed_n_frames# min(1000, processed_n_frames)
        keep_in_memory = output_path is None

        with open(rawfile, 'rb') as reader:
            processed_output = movie_processing.process_stack(
                reader,
                height=height,
                width=width,
                n_frames=raw_n_frames,
                start=0,
                end=frames_to_process,
                channel=1,
                output_path=output_path,
                return_stack=keep_in_memory,
                output_dtype=np.uint16,
                apply_gaussian=True,
                processing_params=params,
                oof_dataframe=oof,
                zarr_chunks=(128, 128, 128),
                zarr_shards=(1024, 128, 128),
                ome_zarr_scale={'t': 1.0, 'y': 1.0, 'x': 1.0},
                show_progress=False,
                progress_desc=rawfile.name,
                device='auto',
                overwrite=False
            )

        if output_path is not None:
            zarr_list.append(Path(processed_output))
            zarr_list_file.write_text('\n'.join(str(path) for path in zarr_list) + '\n')
    return zarr_list


if __name__ == "__main__":
    zarr_list = main()