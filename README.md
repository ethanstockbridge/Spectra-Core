# Introduction
Spectra is an artificial intelligence software that can identify the sound of multiple different classes, all at once! Spectra identifies classes, generates sensor data graphs, and more!
Spectra supports environment sensors such as temperature, humidity, light sensors, pretty much anything that is quantifiable!  
Running on a cutting edge, yolo-based backend, this software is quick and efficient.  
Please note that this repository is coded in my free time with no funding, and thus may contain bugs.  
The associated repositories [Core](https://github.com/ethanstockbridge/Spectra-Core) and [UI](https://github.com/ethanstockbridge/Spectra-UI) framework work together, which allows users to easily access the user interface on any browser.

# Requirements:  
* Python 3.7.9: tested & working. (Other versions ***may*** also work, but no guarantee)
* Modules: opencv, numpy, yolo, etc, install with `pip install -r requirements.txt`

# Hardware requirements

- Please note that there are certain hardware requirements that are needed to run the artificial intelligence, such as nvidia cuda. This mostly pertains to yolo and the underlying modules that it uses, so please check out their website for more information  
    - Tested and working on NVIDIA GeForce GTX 1050ti

# Quickstart

## Installation

1) install python 3.7.9 (recommended)
2) pip install -r requirements.txt
3) Edit the variables.py to update the paths of your project
4) py start_server.py

## File & yolo configuration

Begin by cloning this project and creating some of the (missing) following folders below. You will notice that some exist, some don't. The ones that do not exist are for user read/write:

```
- Spectra-Core
    - config
    - ...
    - recorded
        -class1name
            -audio1.wav/mp3/m4a
            -audio2.wav/mp3/m4a
        -class2name
            - ...
        - ...
        -mixed/random (optional)
    - audio (generated, same layout as recorded)
    - images (generated, same layout as recorded)
    - labeled (same layout as recorded)
    - unlabeled (same layout as recorded)
    - yolo
        -runs (generated by yolo)
        -train
            -images
                -image1.png
                -image2.png
            -labels
                -image1.txt
                -image2.txt
        -val (same as layout train, but with validation set)
        - config.yaml
        - yolov8<x>.pt
```

# Training:  
**Step 1:**  
Gather data in the form of audio files. These can be pre-recorded from a phone, tablet, or PC.  
After your files are ready, arrange the files appropriately as shown above (refer to 'recorded').  
For the 'recorded' folder, this contains only the desired class to make labeling easier. Audio files should also be named the appropriate name, such as "class-location-date" (ex: spring_peeper-location1-04052024) to accurately document them.
These files must then be converted into WAV files with a sample rate of 44100. You can use the script `convert_to_wav.py` in utilities to convert audio files to wav format. This will source audio from the recorded folder and generate the audio folder.  
**Step 2:**  
Run the extract_data.py file. This will perform the algorithms required to turn the audio files into image files, which you will use to label the data. The output of the extraction script will be $PROJ_ROOT_DIR/images/ The format will follow the same as $PROJ_ROOT_DIR/audio/. This procress will take a while...  
<img src="./res/spectro_296.jpg" width="300"/>  
**Step 3:**  
(Simple method) Label your data using something like [labelimg](https://pypi.org/project/labelImg/), save off the output files to a folder like: $PROJ_ROOT_DIR/yolo/train/images and $PROJ_ROOT_DIR/yolo/train/labels  
<img src="./res/predicted_296.jpg" width="300"/>  
(Advanced method) An alternative labeling tool you can use is called [anylabeling](https://github.com/vietanhdev/anylabeling), which is what I use. I think that it can offer some easier labeling but you need to postprocess their json labels into txt labels for yolo to train on.
After labeling, I separate my images and labels into /labeled/ and /unlabeled/ for better organization. You must then use the script `/utilities/split_train_val.py` to copy a 70/30 image split (recommended) from the labeled directory into the yolo train and val folders, respectively.
After splitting, you can run the the script `/utilities/anylabeling_json_to_yolo.py` to convert the json labels to txt yolo format to get it ready to train yolo.  
**Step 4:**  
Train yolo:
You should configure your custom config.yaml, I have included mine if you want to use that to train, you must set up the correct paths first though (adding soon). To run yolo you have to first install [yolov8](https://github.com/ultralytics/ultralytics) with python and be able to call `yolo` on the command line, then cd into your yolo folder shown above, and send the command:  
yolo train data=config.yaml
Once downloaded the initial yolo model, you can train offline using this:  
`yolo train model=yolov8n.pt data=config.yaml`  
Additional typical yolo arguments can be added such as `epochs=500`, `lr0=0.001`, etc.

# User interface
This is a work in progress, but I have made a user interface using React that allows real-time identification of calls. Note that you must first complete a training to use the UI (otherwise it's kind of pointless trying to identify without using a model) 
After completion of training, follow the instructions to fetch and install the UI repository:
[Spectra-UI](https://github.com/ethanstockbridge/Spectra-UI)

# License  
This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE.
Please see the LICENSE file for more information  