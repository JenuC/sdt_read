  
import os
import glob
import argparse
import tifffile as tf
import numpy as np
#from sdt_read.read_openscan_sdt import *
from read_openscan_sdt import read_sdt_openscan

def main(folder):
    os.chdir(folder)
    filelist = glob.glob('*.sdt')
    print("Found files:", filelist)

    for filename in filelist:
        data, _, _ = read_sdt_openscan(filename)
        output_filename = filename[:-4] + '_tf.tif'
        tf.imwrite(output_filename, data[0].transpose(2, 0, 1).astype(np.uint16))
        print(f"Processed {filename}: shape {data[0].shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .sdt files to .tif')
    parser.add_argument('folder', help='Path to folder containing .sdt files')
    args = parser.parse_args()
    main(args.folder)