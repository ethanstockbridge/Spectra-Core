
import os

import numpy as np
from PIL import Image

from config.config_manager import ConfigManager
from utilities.audio import Audio
from utilities.spectrogram import spectrogram


def rgb(minimum, maximum, value):
    """Report RGB representation of a value, useful if you want to
    produce a unique color from 0-100, if value is 44, it will be a color

    Args:
        minimum (int): min range
        maximum (int): max range
        value (int): value between

    Returns:
        (int,int,int): rgb representation
    """
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2*(value-minimum) / (maximum-minimum)
    b = max(0,255*(1-ratio))
    b=int(b)
    r = max(0,255*(ratio-1))
    r=int(r)
    g = 255-b-r
    return r,g,b

def spectro_gen(path, Sxx, yolo_train_size, image_gain):
    """Generate a spectrogram given the path and image data

    Args:
        path (str): Path to write the spectrogram from
        Sxx (numpy.array): Sxx
        yolo_train_size (int): Output image size (match yolo)
        image_gain (int): Amount to increase image by (-1=auto)

    Returns:
        PIL.Image: Image result
    """
    config = ConfigManager()
    # Sxx=(Sxx-np.min(Sxx))/(np.max(Sxx)-np.min(Sxx))*255
    Sxx *= image_gain * config["spectrogram"]["default_image_gain"]
    format = config["spectrogram"]["format"]

    # Calculate RGB values for the entire image
    r_values = np.zeros_like(Sxx, dtype=np.uint8)
    g_values = np.zeros_like(Sxx, dtype=np.uint8)
    b_values = np.zeros_like(Sxx, dtype=np.uint8)

    for y in range(len(Sxx)):
        r_values[y], g_values[y], b_values[y] = rgb(0, len(Sxx), y)

    # Apply opacity to strength
    if format == "png": #NOTE:Untested
        val = (Sxx / 255).astype(np.uint8)
        alpha_channel = val  # Using Sxx as alpha channel
        rgba_image = np.stack((r_values, g_values, b_values, alpha_channel), axis=-1)
    if format == "jpg":
        val = (Sxx / 255).astype(np.float32)
        rgba_image = np.stack((r_values * val, g_values * val, b_values * val), axis = -1)

    # Create PIL Image from RGBA array
    rgba_image_int = np.round(rgba_image).astype(np.uint8)
    img = Image.fromarray(rgba_image_int)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.resize((yolo_train_size,yolo_train_size), Image.LANCZOS)

    if path != None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        img.save(path)

    return img


class DataPoint:
    """Each file turns into this container which holds:
    Raw audio data
    spectrogram data
    Classification name(s) in audio
    Split time data
    """

    def __init__(self,names,path):
        """Initialize the datapoint

        Args:
            names (list<str>): List of strings of this class
            path (str): Path to the audio file
        """
        self.__config = ConfigManager()
        classification = self.__config["labels"]["class_names"]
        self.names=names
        self.classification = [0]*len(classification) #get the classification of this data
        for name in self.names:
            self.classification[classification.index(name)]=1
        self.classification=np.array(self.classification)
        self.path=path
        self.filedata=None
        self.audio=None
        self.spectrogram=None
        self.audio_splits=[]
        self.sxxs=[]


    def extract_audio(self):
        """Extract the audio data from the file
        """
        self.audio=Audio(path=self.path)


    def generate_spectrograms(self, save_path=None, force=False):
        """Save the spectrogram

        Args:
            save_path (str, optional): Path to save it to. Defaults to None.
        """
        if save_path==None:
            save_path=self.path
        if self.sxxs is None:
            return
        for i in range(0,len(self.sxxs)):
            save_path_img = save_path.replace(".wav","").replace("audio","images")+f"_{i}"
            if(os.path.exists(save_path_img+".jpg") and force==False):
                continue
            self.generate_spectrogram(self.sxxs[i], save_path_img)

    def generate_spectrogram(self, split, save_path=None):
        """Save the spectrogram

        Args:
            save_path (str, optional): Path to save it to. Defaults to None.
        """
        format = self.__config["spectrogram"]["format"]
        image_save_path = ""
        if format=="png":
            image_save_path=f"{save_path}.png"
        if format=="jpg":
            image_save_path=f"{save_path}.jpg"
        spectro_gen(image_save_path, split, self.__config["yolo"]["size"], self.__config["spectrogram"]["image_gain"])

    def get_audio_splits(self):
        """Split the audio into chunks and append the Sxx
        """
        self.audio_splits = self.audio.split()
        return self.audio_splits

    def generate_sxxs(self):
        self.sxxs = []
        if(self.audio_splits == None):
            self.split_audio()
        for audio in self.audio_splits:
            self.generate_ssx(audio)

    def generate_ssx(self, audio_split):
        fmin = self.__config["audio"]["fmin"]
        fmax = self.__config["audio"]["fmax"]
        s = spectrogram(audio_split, self.audio.fs)
        f,Sxx=s.limit_frequencies(fmin, fmax)
        self.sxxs.append(Sxx)
        return Sxx