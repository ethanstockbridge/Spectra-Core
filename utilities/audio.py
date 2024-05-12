import os
import time
import wave

import numpy as np
import pyaudio
import scipy.io.wavfile as wavfile
from pydub import AudioSegment

from config.config_manager import ConfigManager


class Audio():
    """Holds audio data
    """

    def __init__(self, path = None, audio=b''):
        """Initialize the audio object
        """

        self.__config = ConfigManager()
        self.fs = self.__config["audio"]["sample_rate"]
        self.sample_width = self.__config["audio"]["sample_width"]
        self.time_period = self.__config["audio"]["time_period"]
        self.channels = self.__config["audio"]["channels"]
        self.path=None
        self.audio=audio

        if path!=None:
            # Convert the audio to the desired format
            audio = AudioSegment.from_file(path)
            audio = audio.set_frame_rate(self.__config["audio"]["sample_rate"])  # Set sample rate to 44100Hz
            audio = audio.set_sample_width(self.__config["audio"]["sample_width"])  # Set to 16-bit
            audio = audio.set_channels(self.__config["audio"]["channels"])  # Set to mono
            processed_path = os.path.join(os.path.dirname(path), f'converted_{os.path.basename(path.replace(".","_"))}.wav')
            audio.export(processed_path, format="wav")
            
            try:
                fs, aud = wavfile.read(processed_path)
                self.fs=fs
                self.audio = b''.join(aud)
                os.remove(processed_path)
            except Exception as e:
                print("ERROR: Could not parse audio, check format:")
                print(e)
                print(processed_path)
                exit(1)
            self.path=path

    def __add__(self, other):
        """Add two audios together

        Args:
            other (Audio): other audio

        Returns:
            Audio: combined audio
        """
        if isinstance(other, Audio):
            return Audio(audio=self.audio+other.audio)
        else:
            # Handle the case when 'other' is not an Audio instance
            raise TypeError("Unsupported type for concatenation")

    def split(self):
        """Split the audio file into chunks of size time_period

        Returns:
            list<list<bytes>>: Returns a split list of list of bytes (audio data)
        """
        self.__splits = []
        if(len(self.audio)<self.fs*self.time_period*self.sample_width):
            return [self.audio]
        for x in range(0,len(self.audio),self.fs*self.time_period*self.sample_width):
            self.__splits.append(self.audio[x:x+self.fs*self.time_period*self.sample_width])
        return self.__splits
    
    def play(self):
        """Play the audio clip to the default speaker
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=self.channels, rate=self.fs, output=True)
        audio = np.array(self.audio, dtype='int16')
        # TODO: Find a better fix?

        # Pad the audio to prevent cutout
        blank_duration = 3
        blank_samples = int(blank_duration * self.fs)
        blank_audio = np.zeros(blank_samples, dtype='int16')
        
        combined_audio = np.concatenate((audio, blank_audio))
        
        stream.write(combined_audio)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def save(self, path=None):
        """Save the audio to a given path

        Args:
            path (str, optional): Save path. Defaults to None.
        """
        if not path:
            path = self.path
        print(type(self.audio))
        if self.audio==b'':
            return
        wavefile = wave.open(path, 'wb')
        wavefile.setnchannels(self.__config["audio"]["channels"])
        wavefile.setframerate(self.__config["audio"]["sample_rate"])
        wavefile.setsampwidth(self.__config["audio"]["sample_width"])
        wavefile.writeframes(self.audio)
        wavefile.close()

    def length(self):
        duration=(len(self.audio))/float(self.fs * self.sample_width * self.channels)
        return duration