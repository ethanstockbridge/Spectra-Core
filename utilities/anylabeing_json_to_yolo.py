import json
import os
import shutil


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

def gather_files(path):
    """Gather the json files from the path

    Args:
        path (str): input path to labels

    Returns:
        list<str>: list of json files
    """
    files = []
    gather_files_rec(path, files)
    files_json = [file for file in files if file.find(".json")!=-1]
    return files_json

def gen_keymap(path):
    """Generate the keymap (classes.txt)

    Args:
        path (str): input path to labels

    Returns:
        dict<str:int>: keymap
    """
    files = gather_files(path)
    dict = {}
    for file in files:
        with open(file, "r", encoding='utf-8') as file:
            dat = file.read()
            if(dat == ""):
                continue
            json_dat = json.loads(dat)
            for obj in json_dat["shapes"]:
                label_name = obj["label"]
                if label_name not in dict.keys():
                    dict[label_name] = len(dict.keys())
    return dict

def convert_anylabeling_json_to_yolo(path, keymap=None):
    """Convert anylabel formatted json to yolo

    Args:
        path (str): Input path to labels
        keymap (dict<str:int>, optional): Keymap. Defaults to None.
    """
    if keymap==None:
        keymap = gen_keymap(path)
    files = gather_files(path)
    for file in files:
        yolo_name = file.replace(".json",".txt")
        output = ""
        with open(file, "r", encoding='utf-8') as file:
            dat = file.read()
            if dat=="":
                continue
            json_dat = json.loads(dat)
            image_width = json_dat["imageWidth"]
            image_height = json_dat["imageHeight"]
            assert(image_width==image_height)
            for obj in json_dat["shapes"]:
                label_name = obj["label"]
                if(obj["shape_type"] == "rectangle"):
                    xy,x2y2 = obj["points"]
                    x,y,x2,y2 = xy[0],xy[1],x2y2[0],x2y2[1]
                    x,y,x2,y2 = x/image_width,y/image_width,x2/image_width,y2/image_width
                if(obj["shape_type"] == "polygon"):
                    x,y,x2,y2 = 99999999999,999999999999,-1,-1
                    points = obj["points"]
                    for point in points:
                        x = min(x,point[0])
                        y = min(y,point[1])
                        x2 = max(x2,point[0])
                        y2 = max(y2,point[1])
                    x,y,x2,y2 = x/image_width,y/image_width,x2/image_width,y2/image_width
                x_cen, y_cen, w, h = x+(x2-x)/2, y+(y2-y)/2, x2-x, y2-y
                output+=f"{keymap[label_name]} {x_cen} {y_cen} {w} {h}\n"
        with open(yolo_name, "w") as file:
            file.write(output)
    with open(os.path.join(path,"classes.txt"),"w") as file:
        for key in keymap.keys():
            file.write(key)
            file.write("\n")

def generate_blank_txt(path):
    """Generate the blank txt annotation files for ones that have no labels

    Args:
        path (str): input path to labels
    """
    files = os.listdir(path)
    files_txt = [x.replace(".txt","") for x in files]
    files_images = [x.replace(".jpg","").replace(".png","") for x in files if x.find(".jpg")!=-1 or x.find(".png")!=-1]

    for name in files_images:
        if name not in files_txt:
            with open(os.path.join(path,name+".txt"), 'w') as fp:
                pass

def move_labels(path_images, path_labels):
    """Move all txt labels over to the labels path

    Args:
        path_images (str): input path to images
        path_labels (str): output path to labels
    """
    if not os.path.exists(path_labels):
        os.makedirs(path_labels)
    files = []
    gather_files_rec(path_images, files)
    txts = [x for x in files if x.find(".txt")!=-1]
    for txt in txts:
        shutil.move(os.path.join(txt), os.path.join(path_labels, os.path.basename(txt)))

if __name__ == "__main__":
    for sub in ["train", "val"]:
        path_images = rf"C:\Users\Ethan\Documents\Spectra-Core\yolo\{sub}\images"
        path_labels = rf"C:\Users\Ethan\Documents\Spectra-Core\yolo\{sub}\labels"
        convert_anylabeling_json_to_yolo(path_images)
        generate_blank_txt(path_images)
        move_labels(path_images, path_labels)
