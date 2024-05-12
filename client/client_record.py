import os
import pickle
import sys
import threading
import time
import wave
from math import floor
from threading import Thread

import pyaudio
import zmq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config_manager import ConfigManager
from proto.SpectraMessage_pb2 import ClientDataMsg
from utilities.recorder import Recorder

if os.path.exists("log.txt"):
    os.remove("log.txt")

input_index=-1

def print2(inp):
    open("log.txt","a+").write(f"{time.time()}: {inp}\n")
    print(inp)

class MicRecorder:
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
        self.frames_lock = threading.Lock()  # Initialize a lock for self.frames
        self.filename = filename
        self.recording=False

    def record(self):
        """Record from the default mic

        Args:
            input_index (int): Input index of mic
        """
        p = pyaudio.PyAudio()

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
            with self.frames_lock:
                self.frames.append(data)
        
        # close stream
        stream.stop_stream()
        stream.close()
        p.terminate()

    def read(self):
        """Read recording relative to last reading

        Args:
            read_time (int, optional): Amount of time(s) to read from buffer. Defaults to -1 (reads all)

        Returns:
            list<bytes>: audio
            bool: success
        """
        send_frames = self.frames[:]
        print(len(send_frames))
        with self.frames_lock:
            self.frames = []
        print(len(self.frames))
        return send_frames

    def record_async(self):
        """Record in a new thread

        Args:
            input_index (int, optional): Input index of mic. Defaults to -1.
        """
        self.recording=True
        self.__record_thread = Thread(target=self.record,args=())
        self.__record_thread.setDaemon(True)
        self.__record_thread.start()

    def stop(self):
        """stop recording
        """
        self.recording=False

    def start(self):
        """Start recording

        Args:
            input_index (int, optional): Default device to use to record. Defaults to -1.
        """
        self.record_async()


class ClientAudioSender:
    """Handles recording and socket connections for the client.
    """
    def __init__(self, publish_port, chunk_time):
        self.context = zmq.Context()
        self.__socket_audio = self.context.socket(zmq.PUB)
        self.__socket_audio.bind("tcp://*:" + publish_port)
        self.audio_id = 0
        self.recorder = MicRecorder()
        self.chunk_time = chunk_time

    def start(self):
        self.recorder.start()
        threading.Thread(target=self.send_frames_async, daemon=True).start()

    def send_frames_async(self):
        while True:
            time.sleep(self.chunk_time)
            audio_data = pickle.dumps(self.recorder.read())
            client_message = ClientDataMsg()
            client_message.audio_data = audio_data
            client_message.time = time.time()
            client_message.audio_id = self.audio_id
            try:
                self.__socket_audio.send(client_message.SerializeToString())
                print2(f"Message {self.audio_id} published")
                self.audio_id += 1
            except zmq.error.ZMQError:
                print2("Error: Could be no interface listening... Trying again next frame")


if __name__ == "__main__":
    minutes_restart = 60

    config = ConfigManager()
    publish_port = config["recording"]["remote"]["sender"]["port"]
    chunk_time = config["recording"]["remote"]["chunk_time"]
    recorder_manager = ClientAudioSender(publish_port, chunk_time)
    recorder_manager.start()
    next_restart = time.time() + minutes_restart*60

    while True:
    #     if time.time()>next_restart:
    #         recorder_manager.stop()
    #         recorder_manager.start()
        time.sleep(1)

    # recorder_manager.stop()
    # time.sleep(5)
    # recorder_manager.stop()
    # recorder_manager = ClientAudioSender(publish_port, chunk_time)
    # recorder_manager.listen_and_send()
    # recorder_manager.send_frames()
    # time.sleep(5)
    # recorder_manager.start()
