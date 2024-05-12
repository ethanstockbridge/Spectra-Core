import os
import shutil
from random import random

labeled_path = r"C:\Users\Ethan\Documents\Spectra-Core\labeled"
train_path = r"C:\Users\Ethan\Documents\Spectra-Core\yolo\train"
val_path = r"C:\Users\Ethan\Documents\Spectra-Core\yolo\val"
val_prob = 0.3

def gather_files_rec(path, files):
    """Gather files recursively

    Args:
        path (str): input path to labels
        files (list<str>): files found
    """
    file_list = os.listdir(path)
    for file in file_list:
        fullfile = os.path.join(path,file)
        if os.path.isdir(fullfile):
            gather_files_rec(fullfile, files)
        else:
            files.append(fullfile)

if __name__ == "__main__":
    files = []
    gather_files_rec(labeled_path,files)
    files = [x.split(".")[0] for x in files] #base
    print(files)

    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.makedirs(os.path.join(train_path, "images"))
    if os.path.exists(val_path):
        shutil.rmtree(val_path)
    os.makedirs(os.path.join(val_path, "images"))

    for file in files:
        basename = os.path.basename(file)
        if not os.path.exists(file+".jpg"):
            continue
        empty_label_file = False
        if not os.path.exists(file+".json"):
            # crete blank anno
            with open(file+".json", 'w') as f:
                pass
            empty_label_file=True
        if random()>=val_prob:
            shutil.copy(file+".jpg",os.path.join(train_path, "images", basename+".jpg"))
            shutil.copy(file+".json",os.path.join(train_path, "images", basename+".json"))
        else:
            shutil.copy(file+".jpg",os.path.join(val_path, "images", basename+".jpg"))
            shutil.copy(file+".json",os.path.join(val_path, "images", basename+".json"))
        #remove empty json to prevent errors in the future because anylabeling doesnt like empty json files
        if empty_label_file:
            os.remove(file+".json")