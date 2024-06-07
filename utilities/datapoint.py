
import os

from torch import NoneType

from config.config_manager import ConfigManager
from utilities.audio import Audio
from utilities.image_utils import spectro_gen
from utilities.spectrogram import spectrogram


class DataPoint:
    """ This is a container which holds:
    Raw audio data
    spectrogram data
    Classification name(s) in audio
    Split time data
    """

    def __init__(self,audio_path=None):
        """Initialize the datapoint

        Args:
            path (str): Path to the audio file
        """
        self.__config = ConfigManager()
        self.audio_path=audio_path
        self.filedata=None
        self.audio=None
        self.spectrogram=None
        self.audio_splits=[]
        self.sxxs=[]

    def extract_audio(self, audio_path=None, audio_bytes=None):
        """Extract the audio data from the file
        """
        if audio_path!=None:
            self.audio_path=audio_path
        self.audio=Audio(path=self.audio_path, audio=audio_bytes)

    def generate_spectrograms(self, save_path=None, force=False):
        """Save the spectrogram

        Args:
            save_path (str, optional): Path to save it to. Defaults to None.
        """
        if audio_path==None:
            audio_path=self.audio_path
        if self.sxxs==None:
            self.generate_sxxs()
        for i in range(0,len(self.sxxs)):
            save_path_img = audio_path.replace(".wav","").replace("audio","images")+f"_{i}"
            if(os.path.exists(save_path_img+".jpg") and force==False):
                continue
            self.generate_spectrogram(self.sxxs[i], save_path_img)

    def generate_spectrogram(self, sxx, save_path=None, image_gain=None):
        """Save the spectrogram

        Args:
            save_path (str, optional): Path to save it to. Defaults to None.
        """
        if image_gain==None:
            image_gain=1
        spectro_gen(save_path, sxx, self.__config["yolo"]["size"], image_gain)

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
            self.generate_sxx(audio)

    def generate_sxx(self, audio_split=None):
        fmin = self.__config["audio"]["fmin"]
        fmax = self.__config["audio"]["fmax"]
        if type(audio_split)==NoneType:
            audio_split=self.audio
        s = spectrogram(audio_split, self.audio.fs)
        f,Sxx=s.limit_frequencies(fmin, fmax)
        self.sxxs.append(Sxx)
        return Sxx