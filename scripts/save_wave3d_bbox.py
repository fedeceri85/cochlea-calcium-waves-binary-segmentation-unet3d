import tifffile as tf
import numpy as np
from skimage.measure import regionprops
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

LISTOFFILES = SCRIPT_DIR / 'processed_zarr_files_for3drois.txt'

def process_waves():
    files_to_process = []
    with open(LISTOFFILES, "r") as f:
        for line in f:
            line = line.strip('\n')
            folder = line.strip('1-jumpCorrected.ome.zarr')
            filepath = folder + 'Wave3DRois.tif'
            files_to_process.append(filepath)

    for f in files_to_process:
        bbox_path = Path(f).with_name('3Droi_bboxes.txt')
        if bbox_path.exists() and bbox_path.stat().st_size > 0:
            print("Skipping non-empty output:", bbox_path)
            continue

        print("processing file:", f)
        try:
            tffile = tf.memmap(f, mode='r')
        except FileNotFoundError:
            print("File not found: " + f)
            continue
            
        roi_props = regionprops(tffile)
        bboxs = [x.bbox for x in roi_props]
        labels = [x.label for x in roi_props]

        #Save to file
        with open(bbox_path, "w") as f:
            for label, bbox in zip(labels, bboxs):
                f.write(str(label)+' ' + str(bbox)+'\n')

if __name__ == "__main__":
    process_waves()

        
