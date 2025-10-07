import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import random
import re 

# Initialize the Flask application
app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = 'random_forest (2).pkl'
DATA_PATH = 'tracks.csv' 

# Columns needed for identification and feature engineering (FE)
REQUIRED_COLUMNS = ['name', 'artists', 'duration_ms', 'danceability', 'energy', 'loudness', 
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 
                    'tempo', 'key', 'mode', 'time_signature'] 

# REQUIRED 14 FEATURES (Model's Expectation):
MODEL_FEATURE_NAMES = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 
    'duration_min', 'energy_dance_ratio', 'acoustic_energy_balance', 
    'mood_index', 'complexity'
]

# --- GLOBAL DATA / MODEL OBJECTS ---
model = None
song_data = None
predicted_class_map = {} 


# --- HELPER FUNCTIONS ---

def clean_artist_name(artist_series):
    """Removes brackets and quotes from the string representation of an artist list."""
    return artist_series.astype(str).str.replace(r"^\[\'|\'\]$", '', regex=True).str.replace(r"\'", '', regex=True)


def load_and_preprocess_data():
    global song_data, model
    
    if song_data is not None:
        return song_data
        
    try:
        if not model:
            print("Model is not loaded. Cannot process data.")
            return None
            
        song_data = pd.read_csv(DATA_PATH)
        
        # Verify raw columns needed for FE (Feature Engineering)
        missing_raw = [col for col in REQUIRED_COLUMNS if col not in song_data.columns]
        if missing_raw:
            print(f"CRITICAL ERROR: Missing raw columns in tracks.csv: {missing_raw}")
            return None

        # --- FEATURE ENGINEERING (Recreating the 14 features for the model) ---
        print("Performing feature engineering to match model input (14 features)...")
        
        epsilon = 1e-6 
        
        # 1. duration_min (Requires duration_ms)
        song_data['duration_min'] = song_data['duration_ms'] / 60000.0
        
        # 2. energy_dance_ratio (Requires energy and danceability)
        song_data['energy_dance_ratio'] = song_data['energy'] / (song_data['danceability'] + epsilon)
        
        # 3. acoustic_energy_balance (MOCKED CALCULATION)
        song_data['acoustic_energy_balance'] = (song_data['acousticness'] + song_data['energy']) / 2
        
        # 4. mood_index (MOCKED CALCULATION using valence and energy)
        song_data['mood_index'] = song_data['valence'] * song_data['energy']
        
        # 5. complexity (MOCKED CALCULATION using speechiness and instrumentalness)
        song_data['complexity'] = song_data['speechiness'] + song_data['instrumentalness']
        
        
        # --- PREDICT CLASS FOR ALL SONGS ---
        print("Predicting classes for all songs using the 14 engineered features...")
        features_for_prediction = song_data[MODEL_FEATURE_NAMES].values
        
        song_data['predicted_class_id'] = model.predict(features_for_prediction)
        
        print(f"Data loading and prediction successful. Total songs: {len(song_data)}")
        return song_data
        
    except Exception as e:
        print(f"CRITICAL ERROR during data loading or feature engineering: {e}")
        return None


# --- INITIALIZATION BLOCK ---

try:
    model = joblib.load(MODEL_PATH)
    num_classes = len(model.classes_) if hasattr(model, 'classes_') else 3 
    predicted_class_map = {i: f"Predicted Class {i}" for i in range(num_classes)}
    
    song_data = load_and_preprocess_data()
    
except Exception as e:
    print(f"Initialization failed: {e}")
    model = None
    song_data = None


# --- FLASK ROUTES ---

@app.route('/')
def home():
    """Renders the index.html file from the templates folder."""
    return render_template('index.html')

@app.route('/api/songs', methods=['GET'])
def get_songs():
    """Returns a simplified list of a LIMITED number of songs for the frontend search bar."""
    if song_data is None:
        return jsonify({'error': f"Data not loaded. Check console for missing raw columns: {REQUIRED_COLUMNS}"}), 500
    
    try:
        SAMPLE_SIZE = 100
        
        if len(song_data) > SAMPLE_SIZE:
            sampled_data = song_data.sample(n=SAMPLE_SIZE, random_state=42).copy()
        else:
            sampled_data = song_data.copy()
            
        # 1. Clean the 'artists' column for display
        sampled_data['artists_clean'] = clean_artist_name(sampled_data['artists'])
            
        # 2. Indexing with 'name' and the cleaned 'artists' column
        song_list = sampled_data[['name', 'artists_clean']].rename(columns={'name': 'title', 'artists_clean': 'artist'}).to_dict('records')
        
        print(f"Sent {len(song_list)} songs to the frontend for search.")
        
        return jsonify(song_list)
    
    except Exception as e:
        error_msg = f"An unexpected error occurred in /api/songs: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    """Recommends 5 songs from the same PREDICTED CLASS ID as the selected song."""
    if song_data is None:
        return jsonify({'error': 'Data processing failed. Cannot recommend.'}), 500

    try:
        data = request.get_json(force=True)
        selected_title = data.get('song_title')
        
        selected_song_row = song_data[song_data['name'] == selected_title]
        
        if selected_song_row.empty:
            return jsonify({'error': f"Song '{selected_title}' not found in the dataset."}), 404

        # Get the predicted class ID (this was calculated at startup)
        predicted_class_id = selected_song_row['predicted_class_id'].iloc[0]
        predicted_class_label = predicted_class_map.get(predicted_class_id, f"Unknown Class {predicted_class_id}")
        
        print(f"Selected song assigned to: {predicted_class_label}")

        # Find Recommended Songs
        class_songs = song_data[song_data['predicted_class_id'] == predicted_class_id]
        
        # Exclude the selected song and ensure we are working on a copy
        recommendable_songs = class_songs[class_songs['name'] != selected_title].copy()
        
        # Clean the artist name for the recommendation list output
        recommendable_songs['artists_clean'] = clean_artist_name(recommendable_songs['artists'])
        
        # 1. Select the columns needed
        cols_for_output = recommendable_songs[['name', 'artists_clean']]
        
        # 2. Rename columns for the frontend (title and artist)
        cols_for_output = cols_for_output.rename(columns={'name': 'title', 'artists_clean': 'artist'})
        
        # 3. Sample and convert to list (using reset_index for robust JSON output)
        if len(cols_for_output) > 5:
            recommended_list = cols_for_output.sample(n=5, random_state=1).reset_index(drop=True).to_dict('records')
        else:
            recommended_list = cols_for_output.reset_index(drop=True).to_dict('records')

        
        return jsonify({
            'selected_song': selected_title,
            'predicted_genre': predicted_class_label, 
            'recommendations': recommended_list
        })

    except Exception as e:
        error_msg = f"An internal error occurred during recommendation. Error: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

# --- RUN APPLICATION ---
if __name__ == '__main__':
    print("\nStarting Flask server. Access the frontend at: http://127.0.0.1:5000/")
    
    # NOTE: Since we are using render_template, we rely on the user having 
    # created the templates/index.html folder and file separately.
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)