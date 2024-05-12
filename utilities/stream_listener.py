""" The purpose of this class and file is to record the stream of a client. The waveform
size is set in the config file.
"""
import os
import pickle
import sys
import threading
import time
import wave

import zmq
from flask import Config

from config.config_manager import ConfigManager
from utilities.audio import Audio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config_manager import ConfigManager
from proto.SpectraMessage_pb2 import ClientDataMsg


class StreamListener:
    def __init__(self, dataset_dir, ip, port, sparse_save):
        self.dataset_dir=dataset_dir
        self.ip = ip
        self.port = port
        self.__config = ConfigManager()
        self.sparse_save=sparse_save

        self.i = 0
        self.audio_data = b''
        self.audio_data_lock = threading.Lock()
        self.time_process_wait = 0.5

        self.stop_event = False

        self.context = zmq.Context()
        self.__socket_audio = self.context.socket(zmq.SUB)
        self.__socket_audio.setsockopt_string(zmq.SUBSCRIBE, "")
        self.__socket_audio.connect("tcp://" + self.ip + ":" + self.port)

    def get_data(self):
        data = self.audio_data
        with self.audio_data_lock:
            self.audio_data = b''
        return data

    def receive_audio(self):
        print("Receiver (ZMQ): Starting to receive audio...")
        while not self.stop_event:
            data = self.__socket_audio.recv()
            response = ClientDataMsg()
            response.ParseFromString(data)
            received_audio_data = pickle.loads(response.audio_data)
            if type(received_audio_data)==list:
                received_audio_data=b''.join(received_audio_data)
            if received_audio_data == b'':
                print("Received data but there was nothing (b'')")
            else:
                with self.audio_data_lock:
                    self.audio_data+=received_audio_data
            print("Receiver (ZMQ): Received audio chunk")
        return

    def save_to_original(self, data_save):
        # TODO: Fix this so it saves all chunks. right now somehow its longer than the chunks combined time
        # Read the existing data from the file (if exists)
        existing_data=b''
        filename_original_audio = os.path.join(self.dataset_dir,f"original_audio.{self.__config['audio']['file_extension']}")
        if(os.path.exists(filename_original_audio)):
            with wave.open(filename_original_audio, 'rb') as rf:
                existing_data = rf.readframes(rf.getnframes())
        # Write the existing data and new data back to the file
        # os.path.remove(filename_original_audio)
        with wave.open(filename_original_audio, 'wb') as wf:
            wf.setsampwidth(self.__config["audio"]["sample_width"])  # Set sample width (2 bytes)
            wf.setframerate(self.__config["audio"]["sample_rate"])  # Set frame rate (44100 Hz)
            wf.setnchannels(self.__config["audio"]["channels"])  # Set number of channels (mono)
            wf.writeframes(existing_data)
            wf.writeframes(data_save)
        print("Saved to original audio file")

    def save_audio(self):
        print("Receiver: Starting to save audio...")
        while not self.stop_event:
            time.sleep(self.time_process_wait)  # Save audio every 3 seconds
            if self.audio_data==b'':
                print("Ready to write, but there is no data to write")
            else:
                with self.audio_data_lock:
                    audio=Audio(audio=self.audio_data)
                    print("Audio length: ",audio.length())
                    if audio.length() >= self.__config["audio"]["time_period"]:
                        chunks = audio.split()
                        if len(chunks)>0:
                            data_save = chunks[0]
                            data_store = chunks[1:]
                            # print("-----------------")
                            # print(audio.length())
                            # print(type(data_save))
                            # print(len(data_save))
                            # print("-----------------")
                            # print("Saving wav file!)")
                            """
                            -----------------
                            3.0238095238095237
                            <class 'bytes'>
                            264600
                            -----------------
                            """
                            with wave.open(
                                os.path.join(self.dataset_dir,
                                    f"{self.__config['audio']['file_prefix']}_{self.i}.{self.__config['audio']['file_extension']}"), 'wb') as wf:
                                wf.setsampwidth(self.__config["audio"]["sample_width"])  # Set sample width (2 bytes)
                                wf.setframerate(self.__config["audio"]["sample_rate"])  # Set frame rate (44100 Hz)
                                wf.setnchannels(self.__config["audio"]["channels"])  # Set number of channels (mono)
                                wf.writeframes(data_save)
                            if not self.sparse_save:
                                self.save_to_original(data_save)
                            self.audio_data = b''.join(data_store)
                            print("Receiver: Saved audio chunk")
                        self.i += 1
        return

    def start(self):
        print("Receiver (ZMQ): Starting...")
        threading.Thread(target=self.receive_audio).start()
        threading.Thread(target=self.save_audio).start()

    def stop(self):
        print("Receiver (ZMQ): Stopping...")
        self.stop_event = True

    def __eq__(self,other):
        return self.ip==other.ip and self.port==other.port

if __name__ == "__main__":
    import os

    config=ConfigManager()
    client_port = config["recording"]["remote"]["sender"]["port"]

    # Create a temporary directory for storing test files
    filelocation = os.path.join("stream_listener_test.wav")
    ip = "127.0.0.1"
    ip="192.168.1.109"
    port = str(client_port)

    # Create a StreamListener instance
    receiver = StreamListener(filelocation, ip, port)

    try:
        # Start the receiver
        receiver.start()

        # Let the receiver run for a while
        time.sleep(10)

    finally:
        # Stop the receiver
        receiver.stop()
