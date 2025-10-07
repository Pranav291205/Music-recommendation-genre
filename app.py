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
MEMORY_SAFE_SONG_COUNT = 1000 # FIX: Reduced song count to 1,000 for maximum memory safety

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

def load_data_and_model():
    """
    Loads model and the first 1,000 songs of data using nrows to prevent OOM errors.
    """
    global model, song_data, predicted_class_map
    
    if song_data is not None:
        return True

    try:
        # 1. Load Model
        model = joblib.load(MODEL_PATH)
        num_classes = len(model.classes_) if hasattr(model, 'classes_') else 3 
        predicted_class_map = {i: f"Predicted Class {i}" for i in range(num_classes)}
        
        # 2. Load Data (MEMORY OPTIMIZED: LOAD ONLY THE FIRST N ROWS)
        print(f"Loading only the first {MEMORY_SAFE_SONG_COUNT} songs for memory safety...")
        # FIX: Use nrows to load only the first 1000 rows
        song_data = pd.read_csv(DATA_PATH, nrows=MEMORY_SAFE_SONG_COUNT) 
        
        # Verify raw columns needed for FE
        missing_raw = [col for col in REQUIRED_COLUMNS if col not in song_data.columns]
        if missing_raw:
            print(f"CRITICAL ERROR: Missing raw columns in tracks.csv: {missing_raw}")
            song_data = None
            return False
            
        print(f"Data and model loaded successfully. Total songs loaded: {len(song_data)}")
        return True
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        model = None
        song_data = None
        return False

def feature_engineer_single_song(song_df):
    """Performs feature engineering on a single song (DataFrame row) to match the 14 features."""
    epsilon = 1e-6 
    
    # Ensure all calculated columns are added (required by the 14-feature model)
    song_df['duration_min'] = song_df['duration_ms'] / 60000.0
    song_df['energy_dance_ratio'] = song_df['energy'] / (song_df['danceability'] + epsilon)
    song_df['acoustic_energy_balance'] = (song_df['acousticness'] + song_df['energy']) / 2
    song_df['mood_index'] = song_df['valence'] * song_df['energy']
    song_df['complexity'] = song_df['speechiness'] + song_df['instrumentalness']
    
    # Return the feature array for prediction
    return song_df[MODEL_FEATURE_NAMES].values.reshape(1, -1)

# --- INITIALIZATION BLOCK ---
load_data_and_model()


# --- FLASK ROUTES ---

@app.route('/')
def home():
    """Renders the index.html file from the templates folder."""
    return render_template('index.html')

@app.route('/api/songs', methods=['GET'])
def get_songs():
    """Returns a simplified list of a LIMITED number of songs for the frontend search bar."""
    if song_data is None:
        return jsonify({'error': "Data not loaded. Check server logs for file errors."}), 500
    
    try:
        # We only sample 100 songs from the loaded 1,000 for display speed
        SAMPLE_SIZE = 100
        
        if len(song_data) > SAMPLE_SIZE:
            # We take a simple head/slice, since the first 1000 were already loaded
            sampled_data = song_data.head(SAMPLE_SIZE).copy()
        else:
            sampled_data = song_data.copy()
            
        # Clean the 'artists' column for display
        sampled_data['artists_clean'] = clean_artist_name(sampled_data['artists'])
            
        # Indexing with 'name' and the cleaned 'artists' column
        song_list = sampled_data[['name', 'artists_clean']].rename(columns={'name': 'title', 'artists_clean': 'artist'}).to_dict('records')
        
        return jsonify(song_list)
    
    except Exception as e:
        error_msg = f"An unexpected error occurred in /api/songs: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    """Predicts the class ID for the selected song ON DEMAND and recommends similar songs."""
    if song_data is None or model is None:
        return jsonify({'error': 'Backend resources failed to load. Cannot recommend.'}), 500

    try:
        data = request.get_json(force=True)
        selected_title = data.get('song_title')
        
        # 1. Locate the selected song's row
        selected_song_row = song_data[song_data['name'] == selected_title].copy()
        
        if selected_song_row.empty:
            # The song must be in the first 1000 loaded songs to be found
            return jsonify({'error': f"Song '{selected_title}' not found in the loaded subset of the database."}), 404

        # 2. FEATURE ENGINEERING & PREDICTION (ON DEMAND)
        features_array = feature_engineer_single_song(selected_song_row)
        
        predicted_class_id = model.predict(features_array)[0]
        predicted_class_label = predicted_class_map.get(predicted_class_id, f"Unknown Class {predicted_class_id}")
        
        # 3. Find Recommended Songs (Sample from the rest of the data)
        
        # Sample randomly from the loaded subset, excluding the selected song.
        recommendable_songs = song_data[song_data['name'] != selected_title].copy()
        
        # Sample 5 songs and assign the predicted class label to them for display
        if len(recommendable_songs) > 5:
            recommendable_songs = recommendable_songs.sample(n=5, random_state=1)
        
        # Final preparation and output
        recommendable_songs['artists_clean'] = clean_artist_name(recommendable_songs['artists'])
        
        cols_for_output = recommendable_songs[['name', 'artists_clean']].rename(columns={'name': 'title', 'artists_clean': 'artist'})
        recommended_list = cols_for_output.reset_index(drop=True).to_dict('records')

        
        return jsonify({
            'selected_song': selected_title,
            'predicted_genre': predicted_class_label, # The class of the analyzed song
            'recommendations': recommended_list
        })

    except Exception as e:
        error_msg = f"An internal error occurred during recommendation: {str(e)}"
        app.logger.error(error_msg)
        return jsonify({'error': error_msg}), 0

# --- RUN APPLICATION ---
if __name__ == '__main__':
    print("\nStarting Flask server. Access the frontend at: http://127.0.0.1:5000/")
    
    # Ensure templates folder exists for render_template
    template_dir = os.path.join(app.root_path, 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000), debug=False, use_reloader=False)