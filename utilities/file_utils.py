import os
import re
import wave

from utilities.audio import Audio


def f_to_c(fahrenheit):
    celsius = (fahrenheit - 32) * 5 / 9
    return celsius

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wf:
        # Get the number of frames
        num_frames = wf.getnframes()
        # Get the frame rate (number of frames per second)
        frame_rate = wf.getframerate()
        # Calculate the duration (length) of the WAV file in seconds
        duration = num_frames / float(frame_rate)
        return duration

def get_latest_file_creation_time(directory):
    latest_time = 0
    latest_file = None

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            creation_time = os.path.getctime(filepath)
            if creation_time > latest_time:
                latest_time = creation_time
                latest_file = filepath

    return latest_time


def get_earliest_and_latest_creation_time(directory):
    # Get list of files in the directory
    start_time=0
    if os.path.exists(os.path.join(directory,"start_time.txt")):
        start_time=float(open(os.path.join(directory,"start_time.txt"),"r").read())

    end_time=0
    if os.path.exists(os.path.join(directory,"end_time.txt")):
        end_time=float(open(os.path.join(directory,"end_time.txt"),"r").read())
    if os.path.exists(os.path.join(directory,"original_audio.wav")):
        end_time = start_time+get_wav_duration(os.path.join(directory,"original_audio.wav"))
        open(os.path.join(directory,"end_time.txt"), "w").write(str(end_time))
    else:
        #last resort, go through every file and get last creation time
        end_time=get_latest_file_creation_time(directory)
        open(os.path.join(directory,"end_time.txt"), "w").write(str(end_time))

    return start_time, end_time

def get_chunk_id(filename):
    # Extract the number from the filename using regular expression
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return float('inf')  # Return infinity if no number is found

