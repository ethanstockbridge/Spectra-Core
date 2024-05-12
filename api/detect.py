import os
import sys
import time

from config.config_manager import ConfigManager
from utilities.audio import Audio
from utilities.datapoint import DataPoint
from utilities.file_utils import get_chunk_id
from utilities.sensor_data_parser import SensorDataParser
from utilities.yolo_detector_yolov8 import YoloDetector

__config = ConfigManager()

def custom_print(*args, sep=' ', end='\n', file=sys.stdout, save_path=None, flush=False):
    """Instead of printing, write it to a file so we can reference the log later

    Args:
        sep (str, optional): Write sep. Defaults to ' '.
        end (str, optional): Write end. Defaults to '\n'.
        file (file, optional): File to write to. Defaults to sys.stdout.
        save_path (str, optional): File to write to. Defaults to None.
        flush (bool, optional): Flush. Defaults to False.
    """
    if save_path:
        file = open(save_path,"a")
    output = sep.join(map(str, args)) + end
    file.write(output)
    if flush:
        file.flush()

def parse_recording(audio_path, min_conf):
    """Parse the recording

    Args:
        audio_path (str): Specified recording to parse
        min_conf (float): Minimum confidence to trigger

    Returns:
        bool: Success
    """
    __yolo = YoloDetector()
    save_path = os.path.dirname(audio_path)
    log_path = os.path.join(save_path,"log.txt")
    
    custom_print(os.path.basename(save_path), save_path=log_path)
    start_time=time.time()

    custom_print("Extracting audio...", save_path=log_path)
    datapoint = DataPoint([],audio_path)
    datapoint.extract_audio()

    custom_print("Splitting audio into segments...", save_path=log_path)
    audio_splits = datapoint.get_audio_splits()
    sxx_splits = []
    for i, audio in enumerate(audio_splits):
        custom_print(f"Generating SXX {i+1}/{len(audio_splits)}", save_path=log_path)
        sxx = datapoint.generate_ssx(audio)
        sxx_splits.append(sxx)

    for i, spectro in enumerate(sxx_splits):
        custom_print(f"Generating spectrogram {i+1}/{len(sxx_splits)}", save_path=log_path)
        datapoint.generate_spectrogram(spectro, os.path.join(save_path,f"spectro_{i}"))
    for i, audio in enumerate(audio_splits):
        custom_print(f"Saving audio {i+1}/{len(audio_splits)}", save_path=log_path)
        a = Audio(audio=audio)
        a.save(os.path.join(save_path,f"audio_{i}.wav"))

    files = os.listdir(save_path)
    files = [x for x in files if x.find(".jpg")!=-1 and x.find("spectro")!=-1]
    for i, file in enumerate(files):
        __yolo.plot_predict(os.path.join(save_path,file),
                            os.path.join(save_path,file).replace("spectro", "predicted"),
                            min_conf)
        custom_print(f"Saving detected images... {i+1}/{len(audio_splits)}", save_path=log_path)
    __yolo.save_labels(os.path.join(save_path, "classes.txt"))
    __export_data(save_path)

    custom_print(f"Complete", save_path=log_path)
    end_time=time.time()
    custom_print(f"Took {end_time-start_time:0.2f} seconds to process {datapoint.audio.length():0.2f}s audio clip", 
                 save_path=log_path)
    custom_print(f"Time speedup: {datapoint.audio.length()/(end_time-start_time):0.2f}x", 
                 save_path=log_path)
    custom_print("Detection complete", save_path=log_path)
    return True


def __export_data(dataset_path):
    """Export the data collected from the audio detections and sensor, to a main "data.csv" file
    for the user to post-process

    data.csv
    Time, Species, sensor1, sensor2
    0,"spring peeper",1,2
    1,"spring peeper",2,3
    2,"spring peeper",3,4

    Args:
        dataset_path (str): Path to save data to
    """
    import datetime

    file_classes_txt = os.path.join(dataset_path,"classes.txt")
    file_data_csv = os.path.join(dataset_path,"data.csv")
    file_sensor_readings = os.path.join(dataset_path,"sensor_readings.csv")
    file_start_time = os.path.join(dataset_path,"start_time.txt")

    classes = open(file_classes_txt, "r").read().strip().split("\n")

    SDP = SensorDataParser(file_sensor_readings)
    
    start_time=float(open(file_start_time,"r").read())

    data = {
        'Time': [],
        'Species': [],
    }

    # Gather columns into file
    for sensor_name in SDP.getSensorNames():
        data[sensor_name]=[]

    # Collect data from each of the prediction label files
    audio_length = __config["audio"]["time_period"] #TODO: change to reference __config
    files = os.listdir(dataset_path)
    # find all spectro_*.txt files
    files = [file for file in files if file.find(".txt")!=-1  and file.find("spectro")!=-1]
    files.sort(key=get_chunk_id)
    for file_name in files:
        file_name = os.path.join(dataset_path,file_name)
        with open(file_name, "r", encoding="utf8") as file:
            f_contents = file.read().strip() #read file
            if(f_contents!=""):
                lines = f_contents.split("\n")
                for line in lines:
                    segments = line.split(" ")
                    class_i = segments[0].strip() # get class number
                    file_i = get_chunk_id(os.path.basename(file_name)) # get the number of this file (spectro_3.txt) -> 3
                    # note: we want to start at #_0 because we dont want initial offset
                    # excel formatting unix seconds to date               =(((A18522/60)/60-4)/24)+DATE(1970,1,1)
                    center_call = start_time+((float(segments[1])+0.5*(float(segments[3])))*audio_length + file_i*audio_length)
                    center_call = float(center_call)
                    # print(datetime.datetime.fromtimestamp(center_call))

                    # at each of the call centers, get the sensor data
                    species = classes[int(class_i)]
                    data["Time"].append(str(center_call))
                    data["Species"].append(species)
                    # print(center_call)
                    # print(species)
                    for sensor_name in SDP.getSensorNames():
                        if sensor_name=="Time":
                            continue #dont add time again because we already calculated time of detection
                        data[sensor_name].append(str(SDP.getSensorData(sensor_name, center_call)))

    csv = ""
    csv+=','.join(data.keys())+"\n"
    for i in range(0,len(data["Species"])):
        columns = []
        for column in data.keys():
            columns.append(str(data[column][i]))
        csv+=','.join(columns)+"\n"
    open(file_data_csv,"w").write(csv)
