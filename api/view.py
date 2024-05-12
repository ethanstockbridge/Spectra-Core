import multiprocessing
import os
import random
from datetime import datetime, timedelta
from io import BytesIO
from itertools import cycle
from math import ceil
from zipfile import ZipFile

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Blueprint, jsonify, request, send_file
from matplotlib.ticker import IndexLocator, MaxNLocator
from pytz import timezone

from api.detect import __export_data, parse_recording
from utilities.file_utils import (f_to_c, get_chunk_id,
                                  get_earliest_and_latest_creation_time)
from utilities.media_id_translator import media_id_translator
from utilities.sensor_data_parser import SensorDataParser
from variables import *

COLOR_PRIMARY = '#00B530'
COLOR_BACKGROUND = '#313537'
colors=[
    "#5A64EE",#purple
    "#3DD72B",#bright green
    "#CF4AE9",#pink
    "#AAAE3E",#yellow
    "#4BEADD"#cyan
]

api_view = Blueprint('/api/dataset', __name__)
image_api = "/api/image/"
graph_width=2
graph_height=1

ImageNotFound="/api/image/no_detections_found"


def generate_data(path_dataset):
    """Generate the data.csv file given the class names and possible sensor data
    Then extract that data and pass it back
    {"Time":[0.2,0.4], "Species":["a","b"], "Sensor1":[9,10]}

    Args:
        path_dataset (str): path to data csv

    Returns:
        dict: data
    """
    print(f"Getting data from file {path_dataset}/data.csv")
    path_data_csv = os.path.join(path_dataset,"data.csv")
    if not os.path.exists(path_data_csv):
    # if True:
        # add data to the data.csv
        __export_data(os.path.dirname(path_data_csv))

    dict_data = {
    }

    try:
        lines = open(path_data_csv, "r").read()
        lines = lines.strip().split("\n")
        if len(lines) == 1:
            print("No detections found")
            return None
    except:
        return None

    columns = lines[0].split(",")
    lines = lines[1:]  # do not include column titles
    for line in lines:
        items = line.split(",")
        for i, column_name in enumerate(columns):
            if column_name not in dict_data:
                dict_data[column_name]=[]
            try:
                dict_data[column_name].append(float(items[i]))
            except:
                dict_data[column_name].append(items[i])

    print(f"Got data from file {path_data_csv}:")

    return dict_data

def plot_distribution(path_graph, dict_data):
    """Plot the graph of species vs time.
    Note that this MUST be in a new process because otherwise it will not be allowed
    since flask uses multithreading to handle requests, and the matplotlib uses qt
    which needs to be in the main thread, so starting a new process will fix that

    Args:
        data (dict): species: detection times
        path_graph (str): output path to graph
    """

    time=dict_data["Time"]
    time.sort()
    time_datetime = [datetime.fromtimestamp(t) for t in time]
    dict_data["Time"]=time_datetime
    print(time_datetime[0], time_datetime[-1])

    # extract_dict={"Time":dict_data["Time"], "Species":dict_data["Species"]}
    df = pd.DataFrame(dict_data)

    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.set_facecolor(COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)
    date_formatter = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(date_formatter)
    ax.xaxis.set_major_locator(MaxNLocator(6))

    plt.xlabel('Time', color=COLOR_PRIMARY)
    plt.ylabel('Species', color=COLOR_PRIMARY)

    plt.title(f'Species call distribution ({time_datetime[0].strftime("%m-%d-%Y")})', color=COLOR_PRIMARY)
    if time_datetime[0].strftime("%m-%d-%Y") != time_datetime[-1].strftime("%m-%d-%Y"):
        plt.title(f'Species call distribution ({time_datetime[0].strftime("%m-%d-%Y")}-{time_datetime[-1].strftime("%m-%d-%Y")})', color=COLOR_PRIMARY)

    palette = [COLOR_PRIMARY for unique_label in set(dict_data["Species"])]

    sns.stripplot(data=df, x='Time', y='Species', jitter=True, alpha=0.7, palette=palette)

    plt.gca().spines["left"].set_color(COLOR_PRIMARY)
    plt.gca().spines["right"].set_color(COLOR_PRIMARY)
    plt.gca().spines["top"].set_color(COLOR_PRIMARY)
    plt.gca().spines["bottom"].set_color(COLOR_PRIMARY)
    ax.tick_params(axis='x', colors=COLOR_PRIMARY)
    ax.tick_params(axis='y', colors=COLOR_PRIMARY)

    # plt.show()
    plt.savefig(path_graph, dpi=300, bbox_inches='tight')
    print("Generated distribution graph")


def plot_count_bar_graph(path_graph, dict_species_time):
    """Plot the graph of species vs time.
    Note that this MUST be in a new process because otherwise it will not be allowed
    since flask uses multithreading to handle requests, and the matplotlib uses qt
    which needs to be in the main thread, so starting a new process will fix that

    Args:
        data (dict): species: detection times
        path_graph (str): output path to graph
    """
    species_counts = {species: len(data_points) for species, data_points in dict_species_time.items()}
    species = list(species_counts.keys())
    counts = list(species_counts.values())

    plt.figure()
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_facecolor(COLOR_BACKGROUND)  # Set background color for axis
    fig.set_facecolor(COLOR_BACKGROUND) # Set background color for figure

    plt.bar(species, counts, color=COLOR_PRIMARY)
    plt.gcf().set_facecolor(COLOR_BACKGROUND)  # Set background color

    plt.xlabel('Species', color=COLOR_PRIMARY)
    plt.ylabel('Count', color=COLOR_PRIMARY)
    plt.title('Species counts', color=COLOR_PRIMARY)

    plt.gca().spines["left"].set_color(COLOR_PRIMARY)
    plt.gca().spines["right"].set_color(COLOR_PRIMARY)
    plt.gca().spines["top"].set_color(COLOR_PRIMARY)
    plt.gca().spines["bottom"].set_color(COLOR_PRIMARY)

    plt.tick_params(axis='x', colors=COLOR_PRIMARY)
    plt.tick_params(axis='y', colors=COLOR_PRIMARY)

    plt.savefig(path_graph, dpi=300, bbox_inches='tight')


def plot_sensor_data(path_graph, path_sensors):
    sdp = SensorDataParser(path_sensors)
    # sdp.normalizeTime()

    data = sdp.getData() 
    if data=={}:
        return None

    time=data["Time"]
    time_datetime = [datetime.fromtimestamp(t) for t in time]

    random.shuffle(colors)
    color_cycle=cycle(colors)

    # Creating the first plot with time and temperature
    # fig, ax1 = plt.subplots()
    fig, ax1 = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    i=0
    ax1.set_xlabel('Time', color=COLOR_PRIMARY)
    ax1.tick_params(axis='y', colors=COLOR_PRIMARY)

    for i, item in enumerate(list(data.keys())[1:]):
        ax=None
        if i==0: #layer on the original graph
            ax=ax1
        else: #spawn new subgraph and go ontop of it
            ax=ax1.twinx()
        if i>=1:
            ax.spines['right'].set_position(('outward', 90*(i-1)))  # Move the spine to the right
            ax.spines['right'].set_color(COLOR_PRIMARY)  # Move the spine to the right
        # Plotting temperature
        color = next(color_cycle)
        ax.set_ylabel(item, color=color)
        ax.plot(time_datetime, data[item], label=item, color=color, marker='', linestyle="-")
        # Set the tick positions

        # ax.set_ylim(min(data[item]),max(data[item])) #set y axis range
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0e')) #scientific notation like 3e+14
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d')) #int
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f')) #float
        date_formatter = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(date_formatter)

        # expand y limits if it is less than 2 
        ax.set_ylim(np.min(data[item])-1, np.max(data[item])+1)

        ax.tick_params(axis='x', colors=COLOR_PRIMARY)
        ax.tick_params(axis='y', labelcolor=color, colors=COLOR_PRIMARY)
        ax.set_facecolor(COLOR_BACKGROUND)
        i+=1

    ax.xaxis.set_major_locator(MaxNLocator(6))

    plt.gca().spines["left"].set_color(COLOR_PRIMARY)
    plt.gca().spines["right"].set_color(COLOR_PRIMARY)
    plt.gca().spines["top"].set_color(COLOR_PRIMARY)
    plt.gca().spines["bottom"].set_color(COLOR_PRIMARY)
    
    # Adding title
    plt.title(f'Sensor Data ({time_datetime[0].strftime("%m-%d-%Y")})', color=COLOR_PRIMARY)
    if time_datetime[0].strftime("%m-%d-%Y") != time_datetime[-1].strftime("%m-%d-%Y"):
        plt.title(f'Sensor Data ({time_datetime[0].strftime("%m-%d-%Y")}-{time_datetime[-1].strftime("%m-%d-%Y")})', color=COLOR_PRIMARY)
    # fig.legend(loc='lower right')
    plt.tight_layout()
    # plt.show()
    plt.savefig(path_graph, figsize=(graph_width,graph_height), dpi=300, bbox_inches='tight')


def more_than_x_in_range(float_list):
    # Sort the list of floats
    """
    0 - no frogs or toads can be heard calling
    1 - individuals can be counted, there is space between calls
    2 - calls of individuals can be distinguished but there is some overlapping calls
    3 - full chorus, calls are constant, continuous and overlapping

    as defined by x*2 calls per timespan
    """
    max_count=0
    float_list=[float(x) for x in float_list]
    float_list=sorted(float_list)
    for i in range(len(float_list)):
        count = 1  # Count the current value itself
        current_time = float_list[i]
        # Count subsequent values within 1 second range
        j = i + 1
        while j < len(float_list) and float_list[j] - current_time <= 1:
            count += 1
            j += 1
        # Update max count
        max_count = max(max_count, count)
    return min(ceil(max_count/2),3)

def extract_species_times(dict_data):
    #Turn the "data" (species:[species,species],time:[time,time]) into
    #{species:[time,time],species:[time,time]}
    time_dict = {}
    for species, time_sec in zip(dict_data['Species'], dict_data['Time']):
        if species not in time_dict:
            time_dict[species] = []
        time_dict[species].append(time_sec)
    dict_species_time = {k: v for k, v in time_dict.items()}
    #dict_species_time is {species:[time,time],species:[time,time]}
    return dict_species_time

def generate_counts(dict_data, dict_species_time):
    str_metrics = "Total Counts:\n"
    for species in set(dict_data["Species"]):
        str_metrics+=f'{species}: {len(dict_species_time[species])}\n'

    dict_metrics = {}

    str_metrics += "\nSpectra str_metrics Calculated:\n"
    for species in list(dict_species_time.keys()):
        calculated_intensity = more_than_x_in_range(dict_species_time[species])
        str_metrics+=f'{species}: {calculated_intensity}\n'
        dict_metrics[species]= calculated_intensity

    return str_metrics, dict_metrics


def generate_distribution_graph(path_graph, dict_data):
    """Generate the plot graph, use multiprocessing because matplotlib needs to be in a main thread

    Args:
        data (dict): species:[species,species],time:[time,time]
        path_graph (str): path to output graph
    """
    print(f"Generating graph: {path_graph}")
    if not os.path.exists(path_graph):
        try:
            mp = multiprocessing.Process(target=plot_distribution, args=[path_graph,dict_data])
            mp.start()
            mp.join()
        except:
            return False


def generate_count_bar_graph(path_graph, dict_species_time):
    """Generate the plot graph, use multiprocessing because matplotlib needs to be in a main thread

    Args:
        data (dict): species:[species,species],time:[time,time]
        path_graph (str): path to output graph
    """
    print(f"Generating graph: {path_graph}")
    if not os.path.exists(path_graph):
        try:
            mp = multiprocessing.Process(target=plot_count_bar_graph, args=[path_graph, dict_species_time])
            mp.start()
            mp.join()
        except:
            return False


def generate_sensor_graph(path_graph, path_sensors):
    """Generate the plot graph, use multiprocessing because matplotlib needs to be in a main thread

    Args:
        data (dict): species:[species,species],time:[time,time]
        path_graph (str): path to output graph
    """
    print(f"Generating graph: {path_graph}")
    if not os.path.exists(path_graph):
        try:
            mp = multiprocessing.Process(target=plot_sensor_data, args=[path_graph, path_sensors])
            mp.start()
            mp.join()
        except:
            return False


@api_view.route('/delete_dataset', methods=['GET', 'POST'])
def delete_dataset():
    """Api endpoint to delete a dataset

    Returns:
        dict: Status
    """
    dataset = request.json.get('dataset')  # assuming dataset is sent in the request JSON
    if dataset=="":
        return jsonify({'status':'error','message':'no such dataset exists'})
    else:
        path_dataset = os.path.join(path_datasets,dataset)
        delete_dataset = path_dataset.replace(dataset,f"deleted_{dataset}")
        os.rename(path_dataset,delete_dataset)
        return jsonify({'status':'ok','message':f'dataset {dataset} was deleted'})


def generate_additional_metrics(path_dataset, dict_metrics):
    timezone_offset = -4
    timezone_diff = timedelta(hours=timezone_offset)
    SpectraNotes = open("./SpectraNotes.txt").read()

    get_earliest_and_latest_creation_time(path_dataset)

    start_time = float(open(os.path.join(path_dataset,"start_time.txt"),"r").read())
    start_time = datetime.utcfromtimestamp(start_time)+timezone_diff
    end_time = float(open(os.path.join(path_dataset,"end_time.txt"),"r").read())
    end_time = datetime.utcfromtimestamp(end_time)+timezone_diff
    script=""
    path_sensors = os.path.join(path_dataset,"sensor_readings.csv")
    sdp=SensorDataParser(path_sensors)
    sensorname_temp = [x for x in sdp.getSensorNames() if x.lower().find("temp")!=-1]
    if len(sensorname_temp)>0:
        script+=f"""Average temp: {round(f_to_c(sdp.getAverage(sensorname_temp[0])),1)} c
"""
    for i, species in enumerate(dict_metrics.keys()):
        script+=f"""Species {i+1} = "{species}"
Intensity {i+1} = "{dict_metrics[species]}"
"""
    script+=f"""Notes: {SpectraNotes}
"""
    return script

# def generate_json_editor(path_dataset, dict_metrics):
#     timezone_offset = -4
#     timezone_diff = timedelta(hours=timezone_offset)
#     SpectraNotes = open("./SpectraNotes.txt").read()

#     get_earliest_and_latest_creation_time(path_dataset)

#     start_time = float(open(os.path.join(path_dataset,"start_time.txt"),"r").read())
#     start_time = datetime.utcfromtimestamp(start_time)+timezone_diff
#     end_time = float(open(os.path.join(path_dataset,"end_time.txt"),"r").read())
#     end_time = datetime.utcfromtimestamp(end_time)+timezone_diff
#     script=f"""document.getElementById("stationId").value = "Spectra HQ";
# document.getElementById("collectionDate").value = "{start_time.strftime('%Y-%m-%d')}";
# document.getElementById("StartTime").value = "{start_time.strftime('%I:%M %p').lower()}";
# document.getElementById("EndTime").value = "{end_time.strftime('%I:%M %p').lower()}";    
# document.getElementById("Notes").value = "{SpectraNotes}";    
# """
#     path_sensors = os.path.join(path_dataset,"sensor_readings.csv")
#     sdp=SensorDataParser(path_sensors)
#     sensorname_temp = [x for x in sdp.getSensorNames() if x.lower().find("temp")!=-1]
#     if len(sensorname_temp)>0:
#         script+=f"""document.getElementById("AirTemperature_value").value = "{round(f_to_c(sdp.getAverage(sensorname_temp[0])),1)}"
#         """
#     for i, species in enumerate(dict_metrics.keys()):
#         script+=f"""document.getElementById("Frog & Toad Observation-fields_{i}_FrogWatch_SpeciesId").value = "{species}"
# document.getElementById("Frog & Toad Observation-fields_{i}_FrogWatch_CallIntensity").value = "{dict_metrics[species]}"
# """
#     return script


@api_view.route('/get_dataset', methods=['GET', 'POST'])
def get_dataset():
    """Api endpoint to get a dataset

    Returns:
        dict: graph image, text, images
    """
    dataset = request.json.get('dataset')  # assuming dataset is sent in the request JSON
    if dataset=="":
        return jsonify({'status':'error','message':'no such dataset exists'})
    print("-----------------------------------------------",dataset)

    path_dataset=os.path.join(path_datasets,dataset)
    # audio_path=os.path.join(path_dataset,"original_audio.wav")

    # parse the whole dataset again if we are missing some items
    # files = os.listdir(path_dataset)
    # original_audio=[x for x in files if x[:6]=="audio_"]
    # predicted=[x for x in files if x[:10]=="predicted_"]
    # if len(original_audio)==0 or len(original_audio)!=len(predicted):
    #     print("Parsing recording because they dont match")
    #     parse_recording(audio_path, 0.2)

    path_sensors=os.path.join(path_dataset,"sensor_readings.csv")

    path_distribution_graph = os.path.join(path_dataset,"graph_distribution.jpg")
    path_species_count_graph = os.path.join(path_dataset,"graph_species_count.jpg")
    path_sensors_graph = os.path.join(path_dataset,"graph_sensors.jpg")

    dict_data = generate_data(path_dataset)
    str_metrics = "No species were found"
    dict_metrics = {}
    if dict_data:
        str_metrics, dict_metrics=generate_counts(dict_data, extract_species_times(dict_data))
        if not os.path.exists(path_distribution_graph):
            generate_distribution_graph(path_distribution_graph, dict_data)
        if not os.path.exists(path_species_count_graph):
            generate_count_bar_graph(path_species_count_graph, extract_species_times(dict_data))

    if not os.path.exists(path_sensors_graph):
        generate_sensor_graph(path_sensors_graph, path_sensors)

    graph_distribution_image_id = media_id_translator().create_new_access(path_distribution_graph)
    graph_distribution_image_url=request.url_root+f"{image_api}{graph_distribution_image_id}"

    graph_species_count_image_id = media_id_translator().create_new_access(path_species_count_graph)
    graph_species_count_image_url=request.url_root+f"{image_api}{graph_species_count_image_id}"

    graph_sensors_image_id = media_id_translator().create_new_access(path_sensors_graph)
    graph_sensors_image_url=request.url_root+f"{image_api}{graph_sensors_image_id}"

    images = os.listdir(path_dataset)
    images=[os.path.join(path_dataset,x) for x in images if x.find(".jpg")!=-1 and x!="graph.jpg"]
    predictions=[x for x in images if x.find("predict")!=-1]
    predictions.sort(key=get_chunk_id)

    # TODO: TBD: Implement thumbnails after database impl?
    # thumbnails=[x for x in images if x.find("thumbnail")!=-1]
    # allimages=[x for x in images if x.find("predict")==-1]
    # if len(thumbnails)!=len(fullsize):
        # for image in fullsize:
            # compress_image(image, image.replace("predict","thumbnail"), 2)
            # thumbnails=[x.replace("predict","thumbnail") for x in fullsize]
    
    images=predictions
    images_ids = []
    for x in images:
        images_ids.append(
            request.url_root+f"{image_api}{media_id_translator().create_new_access(x)}"
        )

    additional_metrics = generate_additional_metrics(path_dataset, dict_metrics)

    response_data = {
        'graph_distribution': (graph_distribution_image_url if os.path.exists(path_distribution_graph) else request.url_root+ImageNotFound),
        'graph_count': (graph_species_count_image_url if os.path.exists(path_species_count_graph) else request.url_root+ImageNotFound),
        'graph_sensors': (graph_sensors_image_url if os.path.exists(path_sensors_graph) else request.url_root+ImageNotFound),
        'text': (str_metrics if str_metrics!=None else "No detections found"),
        'images': images_ids,
        'collection_date': os.path.getctime(os.path.join(path_dataset,"start_time.txt")),
        'start_time': get_earliest_and_latest_creation_time(path_dataset)[0],
        'end_time': get_earliest_and_latest_creation_time(path_dataset)[1],
        'additional_metrics': additional_metrics,
    }

    response = jsonify(response_data)

    return response

@api_view.route('/get_dataset_log/<dataset>', methods=['GET', 'POST'])
def get_dataset_log(dataset):
    """Get the dataset log, this is useful when processing a dataset since it writes to a log file

    Args:
        dataset (str): dataset specified

    Returns:
        str: log of the job
    """
    dataset_path = os.path.join(path_datasets, dataset)
    log_path = os.path.join(dataset_path,"log.txt")
    if not os.path.exists(log_path):
        response_data = {'status':'error','message':'No log for this job, this may mean that it is processing'}
        response = jsonify(response_data)
        return response
    log_text = open(log_path,"r").read()
    response = jsonify({'status': 'ok', 'log': log_text})
    return response

@api_view.route('/get_available_datasets', methods=['GET', 'POST'])
def get_datasets():
    """Get a list of databases, disregarding those deleted (prefix "deleted")

    Returns:
        list(str): list of datasets
    """
    print("Getting datasets from api")
    options = os.listdir(path_datasets)
    options = [x for x in options if x.find("deleted")==-1 and x.find(".")==-1]
    response = jsonify({"datasets":options})
    return response

@api_view.route('/download_dataset/<dataset>')
def serve_dataset(dataset):
    """Download the dataset's zip including wav file, images, etc.

    Args:
        dataset (str): specified dataset

    Returns:
        file: zip of the dataset
    """
    print(f"Dataset {dataset} requested for download")
    dataset_path = os.path.join(path_datasets, dataset)

    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        return "Error, could not find dataset in the datasets folder"

    sendingfile = BytesIO()

    with ZipFile(sendingfile, "w") as myzip:
        for file in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file)
            myzip.write(file_path, os.path.basename(file_path))

    sendingfile.seek(0)
    return send_file(
        sendingfile,
        download_name=f'{dataset}.zip',
        as_attachment=True
    )

@api_view.route('/create_dataset/<dataset>', methods=['GET', 'POST'])
def create_dataset(dataset):
    """Create a new dataset given a name and input wav file

    Args:
        dataset (str): Dataset's name

    Returns:
        dict: status
    """
    print(f"Dataset {dataset} requested for creation")
    dataset_path = os.path.join(path_datasets, dataset)

    uploaded_file = request.files["file"]
    min_confidence = float(request.form.get("minConfidence"))

    if not debug and os.path.exists(dataset_path):
        response_data = {'status': 'error', 'message':'Error, dataset with that tag already exists'}
        return jsonify(response_data)
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    audio_path = os.path.join(dataset_path,"original_audio.wav")
    uploaded_file.save(audio_path)
    creation_time = float(request.form.get("startDate"))
    open(os.path.join(dataset_path,"start_time.txt"),"w").write(str(creation_time))

    th = multiprocessing.Process(target=parse_recording,args=(audio_path, min_confidence))
    th.start()

    response_data = {'status': 'ok'}
    return jsonify(response_data)


@api_view.route('/download_dataset_csv/<dataset>')
def serve_dataset_csv(dataset):
    """Serve the csv of the dataset including frequency of noise

    Args:
        dataset (str): specified dataset

    Returns:
        file: csv file (data.csv)
    """
    print(f"Dataset {dataset} requested for download")
    dataset_csv_path = os.path.join(path_datasets, dataset, "data.csv")

    if not os.path.exists(dataset_csv_path):
        return "Error, could not find dataset in the datasets folder"
    
    sendingfile = BytesIO(open(dataset_csv_path,"rb").read())
    sendingfile.seek(0)

    return send_file(
        sendingfile,
        download_name=f'{dataset}_data.csv',
        as_attachment=True
    )