import json
import math
import os
import pandas as pd
import numpy as np

PROCFOLDER = 'processedMovies'
MASKFILE = 'Wave3DRois.tif'
TIFFFILENAME = '1-jumpCorrected-mc.tif'
ALTERNATIVETIFFFILENAME = '1-jumpCorrected.tif'
OOF_LABELS_FILENAME = 'out_of_focus_labels.csv'
PROCESSING_PARAMS_FILENAME = 'processing_parameters.json'

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
def generate_dataset(alldata,tifffile = TIFFFILENAME, alternative_tifffile = ALTERNATIVETIFFFILENAME, oof_labels_filename = OOF_LABELS_FILENAME, 
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
    dfdata.to_csv(os.path.join(local_data_folder, 'dfdata.csv'), index=False)
    return dfdata