from flask import Flask, request, jsonify
from minio import Minio
from minio.error import S3Error
import os
from io import BytesIO


from flask_cors import CORS
import sys
import numpy as np

import speech_recognition as sr


#from alchemyapi import AlchemyAPI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from preprocessing import extract_fbanks,convert_to_wav
from predictions import get_embeddings, get_cosine_distance

from dotenv import load_dotenv

load_dotenv()


access_key = os.getenv('ACCESS_KEY')
secret_key = os.getenv('SECRET_KEY')


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

DATA_DIR = 'data_files/'
THRESHOLD = 0.45

minio_client = Minio(
    'localhost:9000',  
    access_key=access_key,  
    secret_key=secret_key,  
     secure=False
  
)

bucket_name = 'voice-verif'

# Ensure the bucket exists
if not minio_client.bucket_exists(bucket_name):
    print("Creating the bucket...")
    minio_client.make_bucket(bucket_name)
#----------------------------------------------------------------------------------------------------------------#
@app.route('/')
def home():
    return "Welcome to the Flask backend!"
#----------------------------------------------------------------------------------------------------------------#
@app.route('/register-audio', methods=['POST'])
def register_audio():
    public_address = request.args.get('id')
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    print("FILE IS:",file)
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    wav_filename = f"{public_address}.wav"
    embeddings_filename = f"{public_address}_embeddings.npy"

    try:
    
        wav_io = convert_to_wav(file)
        """minio_client.put_object(
            bucket_name,
            wav_filename,
            wav_io,
            length=wav_io.getbuffer().nbytes,
            content_type='audio/wav'
        )"""
        wav_io.seek(0)  
        fbanks = extract_fbanks(wav_io)  
        embeddings = get_embeddings(fbanks) 
        mean_embeddings = np.mean(embeddings, axis=0) 
        embeddings_io = BytesIO()
        np.save(embeddings_io, mean_embeddings)
        embeddings_io.seek(0)

        minio_client.put_object(
            bucket_name,
            embeddings_filename,
            embeddings_io,
            length=embeddings_io.getbuffer().nbytes,
            content_type='application/octet-stream'
        )

        return jsonify({"message": "Audio and embeddings uploaded and converted successfully"}), 201

    except S3Error as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
#----------------------------------------------------------------------------------------------------------------#
@app.route('/verify-audio/<string:public_address>', methods=['POST'])
def verify_audio(public_address):
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        wav_io = convert_to_wav(file)

        fbanks = extract_fbanks(wav_io)
        embeddings = get_embeddings(fbanks)
        embeddings_filename = f"{public_address}_embeddings.npy"
        embeddings_io = BytesIO()
        
        try:
            response = minio_client.get_object(bucket_name, embeddings_filename)
            data = response.read()
            if not data:
                raise ValueError("Received empty data from MinIO")
            embeddings_io.write(data)
            embeddings_io.seek(0)
            stored_embeddings = np.load(embeddings_io)
        except S3Error as e:
            print(f"Error accessing stored embeddings from MinIO: {e}")
            return jsonify({"error": f"Error accessing stored embeddings from MinIO: {e}"}), 500
        except Exception as e:
            print(f"Unexpected error accessing stored embeddings: {e}")
            return jsonify({"error": f"Unexpected error accessing stored embeddings: {e}"}), 500

        distances = get_cosine_distance(embeddings, stored_embeddings)
        print('Mean distances:', np.mean(distances), flush=True)
        positives = distances < THRESHOLD
        positives_mean = np.mean(positives)
        print('Positives mean:', positives_mean, flush=True)
        
        if positives_mean >= 0.65:
            print("Success")
            return jsonify({"message": "SUCCESS","status":200}), 200
        else:
            print("Failure")
            return jsonify({"message": "FAILURE","status":401}), 401

    except Exception as e:
        print(f"General error: {e}")
        return jsonify({"error": f"General error: {e}"}), 500

@app.route('/check-file', methods=['GET'])
def check_file():
    print("IN CHECK FILE ROUTE")
    public_address = request.args.get('publicAddress')
    if not public_address:
        return jsonify({"error": "Public address is required"}), 400

    file_name = f"{public_address}_embeddings.npy"

    try:
        # Check if the file exists
        print("CHEKING IF IT EXISTS")
        objects = minio_client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            if obj.object_name == file_name:
                return jsonify({"exists": True}), 200
        return jsonify({"exists": False}), 200
    except S3Error as e:
        return jsonify({"error": str(e)}), 500
#----------------------------------------------------------------------------------------------------------------#
@app.route('/transcribe-audio', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    print("LETS INITIALIZE RECONIZER OBJECT")
    recognizer = sr.Recognizer()
    try:
        print("LETS CONVERT TO WAV")
        wav_io = convert_to_wav(file) 
        wav_io.seek(0) 
        print("LETS CALL sr.AudioFile(wav_io)")
        audio_data = sr.AudioFile(wav_io)
        with audio_data as source:
            print("LETS CALL recognizer.adjust_for_ambient_noise")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("LETS CALL recognizer.record(source)")
            audio = recognizer.record(source)
            print("LETS CALL recognizer.recognize_google(audio)")
            text = recognizer.recognize_google(audio)
            text = text.lower()
            return jsonify({"text": text}), 200
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the audio/Unknown Error"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Could not request results from Google Speech Recognition service; {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

#----------------------------------------------------------------------------------------------------------------#
"""
alchemy = AlchemyAPI(environment['ALCHEMY_API_KEY'])

@app.route('/fetch-nfts', methods=['GET'])
def fetch_nfts():
    wallet_address = request.args.get('walletAddress')
    if not wallet_address:
        return jsonify({'error': 'Wallet address is required'}), 400

    try:
        nfts_response = alchemy.nft.getNftsForOwner(wallet_address)
        return jsonify(nfts_response)
    except Exception as e:
        print('Error fetching NFTs:', e)
        return jsonify({'error': 'Error fetching NFTs'}), 500"""



        
if __name__ == '__main__':
   
    app.run(debug=True, use_reloader=False)
