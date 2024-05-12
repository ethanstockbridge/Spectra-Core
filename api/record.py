import os
import threading
import time

import numpy as np
import requests
from flask import Blueprint, request

from config.config_manager import ConfigManager
from utilities.audio import Audio
from utilities.datapoint import spectro_gen
from utilities.file_utils import get_chunk_id
from utilities.media_id_translator import media_id_translator
from utilities.spectrogram import spectrogram
from utilities.stream_listener import StreamListener
from utilities.yolo_detector_yolov8 import YoloDetector
from variables import *

api_record = Blueprint('/api/record', __name__)
record_api = "/api/record/"

temppath = "./temp.wav"

__config = ConfigManager()
__yolo = YoloDetector()
fmin = __config["audio"]["fmin"]
fmax = __config["audio"]["fmax"]
channels = __config["audio"]["channels"]
chunk = __config["audio"]["chunk"]
time_period = __config["audio"]["time_period"]
rate = __config["audio"]["sample_rate"]


def remove_previous(dataset_dir, chunk_id):
    previous_id=chunk_id-1
    if previous_id%__config["recording"]["sparse_index"]!=0:
        path_previous_audio_chunk = os.path.join(dataset_dir, f'audio_{previous_id}.{__config["audio"]["file_extension"]}')
        path_previous_spectro_image = os.path.join(dataset_dir, f'spectro_{previous_id}.{__config["spectrogram"]["format"]}')
        path_previous_predicted_image = os.path.join(dataset_dir, f'predicted_{previous_id}.{__config["spectrogram"]["format"]}')
        try:
            if os.path.exists(path_previous_audio_chunk):
                os.remove(path_previous_audio_chunk)
            if os.path.exists(path_previous_spectro_image):
                os.remove(path_previous_spectro_image)
            if os.path.exists(path_previous_predicted_image):
                os.remove(path_previous_predicted_image)
        except Exception as e:
            print(e)
            return

""" Process new audio in the dataset
"""
class LiveYoloRunner():
    def __init__(self, dataset_dir, detection, gain, min_conf, sparse_save):
        self.time_keepalive = time.time()
        self.detection=detection
        self.gain=gain
        self.min_conf=min_conf
        self.sparse_save=sparse_save
        self.processed = []
        self.dataset_dir=dataset_dir
        self.alive = True
    def start(self):
        self.alive = True
        self.process()
    def process(self):
        while(True):
            #find which files need processing
            current_audios = os.listdir(self.dataset_dir)
            current_audios = [x for x in current_audios if x[0:6]=="audio_"]
            need_processing = [x for x in current_audios if x not in self.processed]
            if len(need_processing)>0:
                print(f"Audio files that need processing: \t\t {need_processing}")
                self.time_keepalive = time.time()
                target_audio = os.path.join(self.dataset_dir,need_processing[0])
                chunk_id = get_chunk_id(os.path.basename(target_audio))
                processFrames(self.dataset_dir, target_audio, self.detection,self.gain,chunk_id,self.min_conf)
                self.processed.append(os.path.basename(target_audio))
                if self.sparse_save:
                    remove_previous(self.dataset_dir, chunk_id)
            if(self.alive==False and len(need_processing)==0):
                return #now we are done processing and no more recordings, so stop.
            time.sleep(0.5)
    def update_params(self,detection, gain, min_conf):
        self.detection=detection
        self.gain=gain
        self.min_conf=min_conf
    def status(self):
        return self.alive
    def stop(self):
        self.alive=False

""" Manage audio processing
"""
class RemoteProcessManager():
    def __init__(self, dataset_dir, detection, gain, min_conf, sparse_save):
        self.detection=detection
        self.gain=gain
        self.min_conf=min_conf
        self.dataset_dir=dataset_dir
        self.sparse_save=sparse_save
        self.id=0
        self.scanned_files = []
        self.unprocessed_files = []
        self.audio_processor = None
        self.live_yolo = LiveYoloRunner(dataset_dir, detection, gain, min_conf, sparse_save)
        self.time_keepalive = time.time()
        self.timeout = 10

    def initialize_processor(self, processor):
        self.audio_processor = processor

    def __eq__(self, other):
        return self.dataset_dir==other.dataset_dir

    def keep_alive(self):
        self.time_keepalive=time.time()
        try:
            if not self.live_yolo.status():
                self.live_yolo = LiveYoloRunner(self.dataset_dir, self.detection, self.gain, self.min_conf, self.sparse_save)
                self.live_yolo_process = threading.Thread(target=self.live_yolo.start(), args=(), daemon=True)
                self.live_yolo_process.start()
        except Exception as e:
            print(f"RemoteProcessManager: {e}")
            self.live_yolo = LiveYoloRunner(self.dataset_dir, self.detection, self.gain, self.min_conf, self.sparse_save)
            self.live_yolo_process = threading.Thread(target=self.live_yolo.start(), args=(), daemon=True)
            self.live_yolo_process.start()

    def kill_smart(self):
        # stops stream listener but keeps processing the remainder of the audio files
        while(True):
            if self.time_keepalive+self.timeout<time.time():
                if self.audio_processor:
                    self.audio_processor.stop()
                self.live_yolo.stop()
                return
            time.sleep(1)

    def start(self):
        if self.audio_processor:
            self.audio_processor.start()
        threading.Thread(target=self.kill_smart, args=()).start()
        self.keep_alive()
        self.live_yolo_process = threading.Thread(target=self.live_yolo.start(), args=(), daemon=True)
        self.live_yolo_process.start()

    def most_recent(self):
        files = os.listdir(self.dataset_dir)
        spectros=[x for x in files if x[:10]=="predicted_"]
        if spectros==[]:
            return None
        spectros.sort(key=get_chunk_id)
        return os.path.join(self.dataset_dir,spectros[-1])

    def update_params(self, detection, gain, min_conf):
        self.detection=detection
        self.gain=gain
        self.min_conf=min_conf
        self.live_yolo.update_params(detection, gain, min_conf)


# Initialize the audio thread
RPMs = []

def parse_request(dataset_dir, detection, gain, min_conf, sparse_save, audio_processor, ip=None, port=None):
    RPM = RemoteProcessManager(dataset_dir, detection, gain, min_conf, sparse_save)
    if audio_processor=="remote":
        RPM.initialize_processor(StreamListener(dataset_dir, ip, port, sparse_save))
    if RPM not in RPMs:
        #start up the receiver
        RPMs.append(RPM)
        RPM.start()
        return {}
    else:
        # check if we can process anything
        RPM = RPMs[RPMs.index(RemoteProcessManager(dataset_dir, detection, gain, min_conf, sparse_save))]
        RPM.keep_alive()
        RPM.update_params(detection, gain, min_conf)
        most_recent = RPM.most_recent()

        # turn the most recent image into an ID
        image_id=-1
        if most_recent != None:
            if(detection):
                image_id = media_id_translator().create_new_access(most_recent.replace("spectro","predicted"))
            else:
                image_id = media_id_translator().create_new_access(most_recent)

        labels_path = os.path.join(dataset_dir, 'classes.txt')
        if not os.path.exists(labels_path):
            __yolo.save_labels(labels_path)
        
        # only add image if there is an image to show
        response_data = {}
        if image_id!=-1:
            response_data = {
                    'image': image_id,
                }

        ################# extract any sensor data (if any)
        weather_station_ip = request.form.get("weatherStationApi")
        if weather_station_ip!="":
            weather_station_response = log_sensor_data(dataset_dir, weather_station_ip)
            response_data["sensorReport"]=weather_station_response

        # find most recent file
        return response_data


@api_record.route("parse_remote_recording/<dataset_id>", methods=['GET', 'POST'])
def parse_remote_recording(dataset_id):
    """Process passback frames from the remote web server, concat audio files if necessary

    Args:
        dataset_id (str): ID of dataset (name)

    Returns:
        dict: resulting image id
    """
    if dataset_id.find(".")!=-1:
        return {"Error, cannot contain '.'"}
    if dataset_id.find("/")!=-1:
        return {"Error, cannot contain '/'"}
    dataset_dir = os.path.join(path_datasets,dataset_id)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    ip = str(request.form.get("remoteIP"))
    port = str(request.form.get("remotePort"))
    detection = bool(request.form.get("detection"))
    gain = float(request.form.get("gain"))
    min_conf = float(request.form.get("minConf"))
    sparse_save=request.form.get("noSave")
    
    dataset_dir = os.path.join(path_datasets,dataset_id)

    return parse_request(dataset_dir, detection, gain, min_conf, sparse_save, "remote", ip, port)


#TODO: have this use similar yolo scanner thing so we can async processing the images
@api_record.route("parse_local_recording/<dataset_id>", methods=['GET', 'POST'])
def parse_local_recording(dataset_id):
    """Process passback frames from the web server, concat audio files if necessary

    Args:
        dataset_id (str): ID of dataset (name)

    Returns:
        dict: resulting image id
    """
    if dataset_id.find(".")!=-1:
        return {"Error, cannot contain '.'"}
    if dataset_id.find("/")!=-1:
        return {"Error, cannot contain '/'"}
    dataset_dir = os.path.join(path_datasets,dataset_id)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    ################## extract audio blob
    detection = bool(request.form.get("detection"))
    gain = float(request.form.get("gain"))
    chunk_id = int(request.form.get("chunkID"))
    min_conf = float(request.form.get("minConf"))
    sparse_save=request.form.get("noSave")

    path_audio_blob = os.path.join(dataset_dir, "incoming_blob.wav")
    path_audio_chunk = os.path.join(dataset_dir, f"audio_{chunk_id}.wav")
    audio_blob = request.files['audio']
    audio_blob.save(path_audio_blob)

    # ################## concat previous incomplete audio chunk (if needed)
    curr_audio=Audio(path=path_audio_blob)
    combined_audio=None

    # combine with leftover audio if it exists
    pre_audio_path = os.path.join(dataset_dir, "pre_audio.wav")
    if os.path.exists(pre_audio_path):
        pre_audio=Audio(path=pre_audio_path)
        combined_audio = pre_audio+curr_audio
        os.remove(pre_audio_path)
    else:
        combined_audio=curr_audio

    # check length
    if combined_audio.length() < __config["audio"]["time_period"]:
        print(f"Less than {__config['audio']['time_period']} seconds, saving chunks")
        combined_audio.save(path=pre_audio_path)
    else:
        splits = combined_audio.split()
        audio_full_chunk = splits[0]
        audio_partial_chunk = splits[1:]
        # save full chunk
        Audio(audio=audio_full_chunk).save(path=path_audio_chunk)
        if len(splits)>1:
            # save remainder
            Audio(audio=b''.join(audio_partial_chunk)).save(path=pre_audio_path)
    ##################

    ################## remove previous dataa
    full_audio_path = os.path.join(dataset_dir,"original_audio.wav")

    if os.path.exists(full_audio_path):
        prev_full_audio = Audio(full_audio_path)
        full_audio = prev_full_audio + curr_audio
        full_audio.save(full_audio_path)
    else:
        curr_audio.save(full_audio_path)
    ##################

    return parse_request(dataset_dir, detection, gain, min_conf, sparse_save, "local")


def processFrames(dataset_dir, path_audio_blob, detection, gain, chunk_id, min_conf):

    if not os.path.exists(path_audio_blob):
        return
    audio = Audio(path=path_audio_blob)
    if not os.path.exists(os.path.join(dataset_dir, 'start_time.txt')):
        open(os.path.join(dataset_dir, 'start_time.txt'),"w").write(str(time.time()))

    ################## perform spectrogram operations
    audio_bytes = np.frombuffer(audio.audio, dtype=np.int16)
    s = spectrogram(audio_bytes, audio.fs)
    f, Sxx=s.limit_frequencies(fmin, fmax)

    # precheck audio gain
    failed = False
    try:
        res = float(gain)
        if(res<0):
            failed=True
    except:
        failed=True
    if failed:
        return -1

    # if path_audio_blob!=os.path.join(dataset_dir,f"audio_{chunk_id}.wav"):
        # audio.save(os.path.join(dataset_dir,f"audio_{chunk_id}.wav"))
    path_spectro_image = os.path.join(dataset_dir,f'spectro_{chunk_id}.{__config["spectrogram"]["format"]}')
    path_predicted_image = os.path.join(dataset_dir,f'predicted_{chunk_id}.{__config["spectrogram"]["format"]}')
    spectro_gen(path_spectro_image, Sxx, __config["yolo"]["size"], float(gain))

    image_id = None
    if(detection):
        print("------------ starting prediction")
        __yolo.plot_predict(path_spectro_image, path_predicted_image, min_conf)
        image_id = media_id_translator().create_new_access(os.path.join(dataset_dir,f"predicted_{chunk_id}.jpg"))
        print("------------ done prediction")
    else:
        image_id = media_id_translator().create_new_access(path_spectro_image)

    labels_path = os.path.join(dataset_dir, 'classes.txt')
    if not os.path.exists(labels_path):
        __yolo.save_labels(labels_path)

    return image_id

def log_sensor_data(dataset_dir, weather_station_ip):
    if "http://" not in weather_station_ip:
        weather_station_ip="http://"+weather_station_ip
    weather_station_ip = weather_station_ip+"/sensors"
    sensor_file = os.path.join(dataset_dir,"sensor_readings.csv")

    try:
        sensor_data = requests.get(weather_station_ip, verify=False, timeout=2).json()
    except:
        print("Could not get weather from the api")
        return "Could not get weather from the weather station API"

    # For testing purposes:
    # sensor_data=[{"name":"Temperature","value":67.82+random.randint(0,10),"units":"F","units_long":"Fahrenheit"},
    #              {"name":"Humidity","value":70+random.randint(0,10),"units":"%","units_long":"Percent"}]

    f = None
    if not os.path.exists(sensor_file): # if the column names do not exist
        f = open(sensor_file, "a")
        columns=["Time"]
        for sensor in sensor_data:
            print(sensor)
            columns.append(sensor["name"]+f" ({sensor['units']})")
        open(sensor_file, "w").write(",".join(columns))
        f.write("\n")
    else: #otherwise, just open the file as normal
        f = open(sensor_file, "a")
    # now log down the values of each column
    f.write(str(time.time())+',')
    for i, sensor in enumerate(sensor_data):
        f.write(str(sensor["value"]))
        if i!=len(sensor_data)-1:
            f.write(",")
    f.write("\n")

    # create return value for the user UI
    string_sensors = ""
    for sensor in sensor_data:
        string_sensors+=f"{sensor['name']}: {sensor['value']}{sensor['units']}\n"
    return string_sensors