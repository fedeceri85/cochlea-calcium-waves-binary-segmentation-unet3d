import json
import math
import os
from pathlib import Path
import pandas as pd
import numpy as np

PROCFOLDER = 'processedMovies'
MASKFILE = 'Wave3DRois.tif'
TIFFFILENAME = '1-jumpCorrected-mc.tif'
ALTERNATIVETIFFFILENAME = '1-jumpCorrected.tif'
OOF_LABELS_FILENAME = 'out_of_focus_labels.csv'
PROCESSING_PARAMS_FILENAME = 'processing_parameters.json'
ZARR_FILENAME = '1-jumpCorrected.ome.zarr'
BBOX_FILENAME = '3Droi_bboxes.txt'

def load_all_excels(folder='./experiment_lists', 
                    exclusions = None,
                    inclusions = None,
                    drive = None
                    ):
    '''
    Load all excel files in the specified folder, excluding certain files based on forbidden keywords, and concatenate them into a single DataFrame. The function also
      processes the 'Folder' column to ensure consistent formatting and removes any rows marked for discard.
    '''
    dfs = []
    if (exclusions is not None ) and  (inclusions is not None):
        raise ValueError("Cannot specify both exclusions and inclusions. Please choose one.")
    
    for file in os.listdir(folder):
        if file.endswith('.xlsx'):
            skip = False
            if exclusions is not None:
                for forbidden in exclusions:
                    if forbidden in file:
                        skip = True
                        break
            elif inclusions is not None:
                skip = True
                for allowed in inclusions:
                    if allowed in file:
                        skip = False
                        break
            if not skip:
                df = pd.read_excel(os.path.join(folder, file))
                df['Source_file'] = file.strip('.xlsx')
                dfs.append(df)

    alldata = pd.concat(dfs, ignore_index=True)
    alldata['Folder'] = alldata['Folder'].str.replace('\\', '/')
    #discard discarded
    alldata = alldata[alldata['discard'] != 1]
    #MACOS
    if drive is not None:
        alldata['Folder'] = 'Z' + alldata['Folder'].str[1:]
        alldata['Folder'] = alldata['Folder'].str.replace('Z:', drive)
        
    alldata.dropna(subset=['Folder'], inplace=True)

    #Create unique id for each Independent experiment
    alldata['UniqueID'] = alldata['Source_file'] + alldata['Independent recordings number'].astype(str)
    return alldata


import tqdm
from shutil import copy2
def generate_dataset(
    alldata,
    image_filename=ZARR_FILENAME,
    maskfilename=MASKFILE,
    bbox_filename=BBOX_FILENAME,
    processed_folder=PROCFOLDER,
    local_data_folder='./data',
):
    """Build a manifest for the 3D wave segmentation transforms.

    Each row contains image (OME-Zarr directory), label (instance TIFF),
    and waves (bbox text file), plus experiment metadata for grouped splitting.
    Missing inputs and empty bbox files are reported and skipped: the current
    foreground sampler requires indexed waves. Array shapes and bbox contents
    are checked by the transforms when loaded; movie pixels are not read here.
    Files stay in place. Set local_data_folder=None to disable CSV export.
    Use dfdata.to_dict('records') as the MONAI Dataset input.
    """
    metadata = ['UniqueID', 'Source_file', 'Independent recordings number', 'Folder']
    missing_columns = set(metadata) - set(alldata.columns)
    if missing_columns:
        raise ValueError(f"Missing dataset columns: {sorted(missing_columns)}")

    records = []
    for _, row in tqdm.tqdm(alldata.iterrows(), total=len(alldata)):
        if pd.isna(row['Folder']) or not str(row['Folder']).strip():
            print('Skipping recording with missing Folder')
            continue
        folder = Path(row['Folder']) / processed_folder
        image = folder / image_filename
        label = folder / maskfilename
        waves = folder / bbox_filename
        missing = []
        if not image.is_dir():
            missing.append(str(image))
        for path in (label, waves):
            if not path.is_file():
                missing.append(str(path))
        if missing:
            print(f"Skipping recording; missing inputs: {', '.join(missing)}")
            continue
        if not waves.read_text().strip():
            print(f'Skipping recording; empty wave index: {waves}')
            continue
        record = {key: row[key] for key in metadata}
        record.update(image=str(image), label=str(label), waves=str(waves))
        records.append(record)

    dfdata = pd.DataFrame(records, columns=metadata + ['image', 'label', 'waves'])
    if local_data_folder is not None:
        output_folder = Path(local_data_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        dfdata.to_csv(output_folder / 'dfdata.csv', index=False)
    return dfdata


def generate_conversion_dataset(alldata,tifffile = TIFFFILENAME, alternative_tifffile = ALTERNATIVETIFFFILENAME, oof_labels_filename = OOF_LABELS_FILENAME, 
                     processing_params_filename = PROCESSING_PARAMS_FILENAME, processed_folder = PROCFOLDER, maskfilename = MASKFILE,
                     local_data_folder = './data'):
    # Crawl the folder and check that avg and masks exists, move them to the 'data' folder
    '''
    Generate a dataset by verifying the existence of average, mask, and annotation files for each entry in the provided DataFrame. The function can optionally copy these files to a local data folder, and it returns a new DataFrame containing the relevant file paths and metadata.
    Parameters:
    - alldata: DataFrame containing the original data with folder paths and metadata.
    - tifffile: Path to the TIFF file containing the average image.
    - processed_folder: Path to the folder containing the processed images.
    - oof_labels_filename: Path to the file containing the out-of-focus labels.
    - processing_params_filename: Path to the file containing the processing parameters.
    - maskfilename: Path to the file containing the mask.
    - local_data_folder: Path to the local data folder where the files will be copied.
    Returns:
    - dfdata: DataFrame containing the new file paths and relevant metadata for each entry.
    '''
    dfdata = []
    for index, row in tqdm.tqdm(alldata.iterrows(), total=alldata.shape[0]):
        folder = row['Folder']
        tif_file = os.path.join(folder, PROCFOLDER, tifffile)
        alternative_tif_file = os.path.join(folder, PROCFOLDER, alternative_tifffile)    
        maskfile = os.path.join(folder, PROCFOLDER, maskfilename)
        oof_labels_file = None
        processing_params_file = None

        if not os.path.exists(maskfile):
            print(f"Masks file does not exist: {maskfile}")
            continue

        
        #Check if the file exists
        if not os.path.exists(tif_file):
            if os.path.exists(alternative_tif_file):
                tif_file = alternative_tif_file
            else:
                #Check if oof labels and parameters exist
                tif_file = None
                oof_labels_file = os.path.join(folder, PROCFOLDER, oof_labels_filename)
                processing_params_file = os.path.join(folder, PROCFOLDER, processing_params_filename)

                if (not os.path.exists(oof_labels_file)) or (not os.path.exists(processing_params_file)):
                    print(f"TIFF file does not exist: {tif_file} and alternative TIFF file does not exist: {alternative_tif_file}")
                    continue

        dfdata.append({
            'UniqueID': row['UniqueID'],
            'Source_file': row['Source_file'],
            'Independent recordings number': row['Independent recordings number'],
            'Folder': folder,
            'TIFF_file': tif_file,
            'Maskfile': maskfile,
            'OOF_labels_file': oof_labels_file,
            'Processing_params_file': processing_params_file
        })
        

        
    dfdata = pd.DataFrame(dfdata)
    output_folder = Path(local_data_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    dfdata.to_csv(output_folder / 'conversion_candidates.csv', index=False)
    return dfdata

def _load_split_json(split_json_path):
    if split_json_path is None or not os.path.exists(split_json_path):
        return set(), set()

    with open(split_json_path, 'r') as f:
        split_data = json.load(f)

    if not isinstance(split_data, dict):
        raise ValueError("Split JSON must contain a dictionary.")

    if 'splits' in split_data:
        split_data = split_data['splits']
    elif 'experiments' in split_data:
        split_data = split_data['experiments']

    split_keys = {'train', 'training', 'train_ids', 'validation', 'val', 'validation_ids', 'val_ids'}
    if split_keys.intersection(split_data):
        train_ids = split_data.get(
            'train',
            split_data.get('training', split_data.get('train_ids', []))
        )
        val_ids = split_data.get(
            'validation',
            split_data.get('val', split_data.get('validation_ids', split_data.get('val_ids', [])))
        )
        return set(train_ids), set(val_ids)

    train_ids = []
    val_ids = []
    for experiment_id, split in split_data.items():
        if isinstance(split, dict):
            split = split.get('split')
        if split in ('train', 'training'):
            train_ids.append(experiment_id)
        elif split in ('val', 'validation'):
            val_ids.append(experiment_id)

    return set(train_ids), set(val_ids)

def _desired_validation_count(test_size, n_experiments):
    if n_experiments == 0:
        return 0
    if isinstance(test_size, int):
        if test_size < 0 or test_size > n_experiments:
            raise ValueError("Integer test_size must be between 0 and the number of experiments.")
        return test_size
    if not 0 < test_size < 1:
        raise ValueError("Float test_size must be between 0 and 1.")
    return math.ceil(n_experiments * test_size)

def _json_safe_id(experiment_id):
    if hasattr(experiment_id, 'item'):
        return experiment_id.item()
    return experiment_id

def _save_split_json(split_json_path, train_ids, val_ids, column, test_size, random_state):
    split_folder = os.path.dirname(split_json_path)
    if split_folder:
        os.makedirs(split_folder, exist_ok=True)

    split_data = {
        'column': column,
        'test_size': test_size,
        'random_state': random_state,
        'train': sorted([_json_safe_id(experiment_id) for experiment_id in train_ids], key=str),
        'validation': sorted([_json_safe_id(experiment_id) for experiment_id in val_ids], key=str),
    }

    with open(split_json_path, 'w') as f:
        json.dump(split_data, f, indent=2)

def train_test_split(dfdata, column='UniqueID', test_size=0.2, random_state=42, split_json_path = None, save_split_json=False):
    """Split the dataframe into training and validation
    """

    # Split the dataframe into training and validation

    uniqueIDs = dfdata[column].unique()
    previous_train_ids, previous_val_ids = _load_split_json(split_json_path)
    duplicate_ids = previous_train_ids.intersection(previous_val_ids)
    if duplicate_ids:
        raise ValueError(f"Experiments assigned to both train and validation in JSON: {sorted(duplicate_ids, key=str)}")

    current_ids = set(uniqueIDs)
    previous_train_ids = previous_train_ids.intersection(current_ids)
    previous_val_ids = previous_val_ids.intersection(current_ids)
    previous_ids = previous_train_ids.union(previous_val_ids)
    new_ids = [experiment_id for experiment_id in uniqueIDs if experiment_id not in previous_ids]

    target_val_count = _desired_validation_count(test_size, len(uniqueIDs))
    new_val_count = target_val_count - len(previous_val_ids)
    new_val_count = max(0, min(len(new_ids), new_val_count))

    if len(new_ids) == 0:
        new_train_ids = []
        new_val_ids = []
    elif new_val_count == 0:
        new_train_ids = new_ids
        new_val_ids = []
    elif new_val_count == len(new_ids):
        new_train_ids = []
        new_val_ids = new_ids
    else:
        from sklearn.model_selection import train_test_split
        new_train_ids, new_val_ids = train_test_split(
            new_ids,
            test_size=new_val_count,
            random_state=random_state
        )

    train_idx = previous_train_ids.union(new_train_ids)
    val_idx = previous_val_ids.union(new_val_ids)

    train_data = dfdata[dfdata[column].isin(train_idx)]
    val_data = dfdata[dfdata[column].isin(val_idx)]

    if save_split_json:
        if split_json_path is None:
            raise ValueError("split_json_path is required when save_split_json=True.")
        saved_train_ids, saved_val_ids = _load_split_json(split_json_path)
        _save_split_json(
            split_json_path,
            saved_train_ids.union(train_idx),
            saved_val_ids.union(val_idx),
            column,
            test_size,
            random_state
        ) 

    #Dataframe to list of dictionaries
    train_data = train_data.to_dict('records')
    val_data = val_data.to_dict('records')
    return train_data, val_data

from monai.data import Dataset, DataLoader
def create_datasets_and_loaders(train_data, val_data, train_transforms, val_transforms,
                                batch_size=2, num_workers=0, val_batch_size=1,
                                shuffle_train=True, shuffle_val=False,
                                val_num_workers=None, train_samples_per_movie=1,):
    if val_num_workers is None:
        val_num_workers = num_workers
    if train_samples_per_movie < 1:
        raise ValueError(
            "train_samples_per_movie must be at least 1"
        )

    expanded_train_data = [
        record.copy()
        for record in train_data
        for _ in range(train_samples_per_movie)
    ]

    train_ds = Dataset(
        data=expanded_train_data,
        transform=train_transforms,
    )

    #train_ds = Dataset(data=train_data, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
                              pin_memory=True,persistent_workers=num_workers > 0)
    val_ds = Dataset(data=val_data,transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=shuffle_val,
        num_workers=val_num_workers,
    )

    return train_ds, train_loader, val_ds, val_loader
