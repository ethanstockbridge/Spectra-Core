import time
import wave
from math import floor
from threading import Thread

import pyaudio

from config.config_manager import ConfigManager


class Recorder():
    """Recorder class
    """

    def __init__(self, filename='out_local.wav'):
        """Initialize the live recorder

        Args:
            filename (str): name of file to write to
            input_index (int): device input. Defaults to 0
        """
        super().__init__()
        self.__config = ConfigManager()
        self.start_time = None
        self.frames=[]
        self.total_frames=[]
        self.filename = filename
        self.recording=False
        #read recording
        self.__read_i = 0

    def read_recent(self, read_time):
        """Read last x time of recording

        Args:
            read_time (int, optional): Amount of time(s) to read from buffer. Defaults to -1.

        Returns:
            list<bytes>: audio
            bool: success
        """
        new_read = floor((read_time*self.__config["audio"]["sample_rate"])/self.__config["audio"]["chunk"])
        if(len(self.frames)<new_read):
            return False
        self.__read_i = new_read
        return self.frames[len(self.frames)-new_read:]

    def read(self, read_time=-1):
        """Read recording relative to last reading

        Args:
            read_time (int, optional): Amount of time(s) to read from buffer. Defaults to -1 (reads all)

        Returns:
            list<bytes>: audio
            bool: success
        """
        prev_read = self.__read_i
        if(read_time==-1):
            self.__read_i = len(self.frames)
            return self.frames[prev_read:self.__read_i]
        
        new_read = floor((read_time*self.__config["audio"]["sample_rate"])/self.__config["audio"]["chunk"])
        if(len(self.frames)<new_read+prev_read):
            return False
        self.__read_i = new_read
        return self.frames[prev_read:prev_read+new_read]

    def record_async(self,input_index=-1):
        """Record in a new thread

        Args:
            input_index (int, optional): Input index of mic. Defaults to -1.
        """
        self.recording=True
        self.__record_thread = Thread(target=self.record,args=(input_index,))
        self.__record_thread.setDaemon(True)
        self.__record_thread.start()

    def stop(self):
        """stop recording
        """
        self.recording=False

    def record(self, input_index):
        """Record from the default mic

        Args:
            input_index (int): Input index of mic
        """
        p = pyaudio.PyAudio()

        self.frames=self.total_frames=[]

        format = None
        if self.__config["audio"]["format"] == "pyaudio.paInt16":
            format = pyaudio.paInt16

        stream = p.open(format=format,
                        channels=self.__config["audio"]["channels"],
                        input_device_index=input_index,
                        rate=self.__config["audio"]["sample_rate"],
                        input=True,
                        frames_per_buffer=self.__config["audio"]["chunk"])
        self.start_time = time.time()
        while self.recording:
            data = stream.read(self.__config["audio"]["chunk"])
            self.total_frames.append(data)
        
        # close stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.save()


    def save(self):
        """Save all audio data to disk
        """
        p = pyaudio.PyAudio()
        wavefile = wave.open(self.filename, 'wb')
        wavefile.setnchannels(self.__config["audio"]["channels"])
        wavefile.setframerate(self.__config["audio"]["sample_rate"])
        format = None
        if self.__config["audio"]["format"] == "pyaudio.paInt16":
            format = pyaudio.paInt16
        wavefile.setsampwidth(p.get_sample_size(format))
        wavefile.writeframes(b''.join(self.total_frames))
        wavefile.close()
