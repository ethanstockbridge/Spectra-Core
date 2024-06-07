import os
import time

from config.config_manager import ConfigManager
from utilities.datapoint import DataPoint

# to overwrite old spectrograms, change this to true:
force = False

def gather_audio(audio):
    """Gather the files from the 'audio' folder

    Returns:
        list<DataPoint>: List of the found data
    """
    data = []
    folders = os.listdir(audio)
    for folder in folders:
        files = os.listdir(os.path.join(audio, folder))
        for file in files:
            if "wav" in file:
                data.append(DataPoint(os.path.join(audio, folder, file)))
    return data


if __name__ == "__main__":
    
    config = ConfigManager()

    # try:
    #     shutil.rmtree(config["paths"]["imagepath"])
    # except:
    #     pass

    print("Collecting data")
    start_time = time.time()
    data_points = gather_audio(config["paths"]["audio"], config["labels"]["class_name"])
    end_time = time.time()
    print(f"Data collection took {end_time-start_time:0.1f} seconds")

    print("Extracting data")
    start_time = time.time()
    for singleton in data_points:
        print(singleton.path)
        singleton.extract_audio()
        print("Splitting audio and generating SXX")
        singleton.generate_sxxs()
        print("Saving spectrograms")
        singleton.generate_spectrograms(force=force)
    end_time = time.time()
    print(f"Extraction took {end_time-start_time:0.1f} seconds")

    print("Audio has been converted into images. Proceed to label the images (using something like labelImg)")
