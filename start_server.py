import os

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from api.record import api_record
from api.view import api_view
from utilities.media_id_translator import media_id_translator
from variables import *

app = Flask(__name__)

app.register_blueprint(api_view, url_prefix='/api/dataset')
app.register_blueprint(api_record, url_prefix='/api/record')

CORS(app)

###################################### MAIN GETTERS ################################

@app.route('/api/audio/<audio_id>')
def serve_audio(audio_id):
    print("Serving audio")
    image_path = media_id_translator()[audio_id]
    print(image_path)
    audio_path = image_path.replace(".jpg",".wav").replace("specto","audio").replace("predicted","audio")
    if not os.path.exists(audio_path):
        return "Error"
    return send_file(audio_path,
                     mimetype='audio/vnd.wav')

@app.route('/api/image/<image_id>')
def serve_image(image_id):
    image_path = media_id_translator()[image_id]
    return send_file(image_path, mimetype='image/gif',)

@app.route('/api/full_audio/<dataset>')
def serve_full_audio(dataset):
    if dataset not in os.listdir(path_datasets):
        return "Error, could not find dataset in the datasets folder"
    audio_path = os.path.join(path_datasets, dataset, "original_audio.wav")
    if os.path.exists(audio_path):
        return send_file(audio_path,
                     mimetype='audio/vnd.wav')
    else:
        return jsonify({"Error": 404})

###################################### DOWNLOADERS ################################


if __name__ == "__main__":
   app.run(debug=True, host='127.0.0.1') # Start with localhost IP