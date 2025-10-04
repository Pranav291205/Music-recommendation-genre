# app.py
from flask import Flask, request, jsonify, render_template 
try:
    from flask_cors import CORS
    cors_available = True
    print("CORS support enabled")
except ImportError:
    cors_available = False
    print("CORS support disabled - flask-cors not installed")

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics.pairwise import cosine_similarity 
import os
import traceback
import ast
import gc

app = Flask(__name__)
if cors_available:
    CORS(app)

# Global variables for the model and data
scaler = None
tracks_df = None
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

def load_data_and_train_model():
    global scaler, tracks_df
    
    print("Starting optimized data loading process...")
    
    # Force garbage collection first
    gc.collect()
    
    try:
        if os.path.exists('tracks.csv'):
            print("Loading tracks.csv with memory optimization...")
            
            # Load only necessary columns to save memory
            usecols = ['name', 'artists', 'popularity'] + feature_columns
            tracks_df = pd.read_csv('tracks.csv', on_bad_lines='skip', usecols=usecols)
            print(f"Dataset loaded with {len(tracks_df)} tracks")
            
            # Reduce memory usage by converting to appropriate data types
            tracks_df = tracks_df.dropna(subset=['name', 'artists'])
            tracks_df = tracks_df[tracks_df['name'].str.len() > 0]
            tracks_df = tracks_df[tracks_df['artists'].str.len() > 0]
            
            # Sample a smaller dataset for deployment
            tracks_df = tracks_df.head(5000)  # Reduced from 20,000 to 5,000
            
            # Clean up artist names
            def safe_parse_artists(artist_str):
                try:
                    if pd.isna(artist_str):
                        return 'Unknown Artist'
                    artist_str = str(artist_str).replace('"', "'")
                    artists = ast.literal_eval(artist_str)
                    if isinstance(artists, list) and len(artists) > 0:
                        return str(artists[0])
                    else:
                        return 'Unknown Artist'
                except:
                    artist_str = str(artist_str).strip("[]'\"")
                    return artist_str if artist_str else 'Unknown Artist'
            
            tracks_df['artist'] = tracks_df['artists'].apply(safe_parse_artists)
            
            # Drop the original artists column to save memory
            tracks_df = tracks_df.drop('artists', axis=1)
            
            # Scale features
            scaler = StandardScaler()
            feature_array = scaler.fit_transform(tracks_df[feature_columns])
            
            # Store as numpy array instead of dataframe column to save memory
            tracks_df['features_scaled'] = [arr for arr in feature_array]
            
            # Force garbage collection
            del feature_array
            gc.collect()
            
            print(f"Optimized data loading completed with {len(tracks_df)} tracks")
            
        else:
            print("tracks.csv not found. Creating lightweight demo data...")
            create_lightweight_demo_data()
            
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Creating lightweight demo data as fallback...")
        create_lightweight_demo_data()

def create_lightweight_demo_data():
    """Create minimal demo data to save memory"""
    global scaler, tracks_df
    
    print("Creating lightweight demo data...")
    
    np.random.seed(42)
    n_samples = 1000  # Reduced from 500 to 1000
    
    # Minimal song data
    real_songs = [
        {"name": "Blinding Lights", "artist": "The Weeknd"},
        {"name": "Shape of You", "artist": "Ed Sheeran"},
        {"name": "Dance Monkey", "artist": "Tones and I"},
        {"name": "Bad Guy", "artist": "Billie Eilish"},
        {"name": "Watermelon Sugar", "artist": "Harry Styles"},
        {"name": "Don't Start Now", "artist": "Dua Lipa"},
        {"name": "Circles", "artist": "Post Malone"},
        {"name": "Levitating", "artist": "Dua Lipa"},
        {"name": "Stay", "artist": "The Kid LAROI"},
        {"name": "Good 4 U", "artist": "Olivia Rodrigo"},
    ]
    
    demo_data = []
    for i in range(n_samples):
        if i < len(real_songs):
            song_info = real_songs[i]
        else:
            prefixes = ['Midnight', 'Summer', 'Electric', 'Golden', 'Neon']
            suffixes = ['Dreams', 'Love', 'Fire', 'Waves', 'Sky']
            artists = ['Taylor Swift', 'Ariana Grande', 'Bruno Mars', 'Drake', 'Rihanna']
            
            song_name = f"{np.random.choice(prefixes)} {np.random.choice(suffixes)}"
            artist_name = np.random.choice(artists)
            song_info = {"name": song_name, "artist": artist_name}
        
        # Simple feature generation
        features = {
            'danceability': np.random.uniform(0.5, 0.9),
            'energy': np.random.uniform(0.5, 0.9),
            'loudness': np.random.uniform(-10, -5),
            'speechiness': np.random.uniform(0.02, 0.1),
            'acousticness': np.random.uniform(0.1, 0.6),
            'instrumentalness': np.random.uniform(0.0, 0.3),
            'liveness': np.random.uniform(0.1, 0.4),
            'valence': np.random.uniform(0.4, 0.8),
            'tempo': np.random.uniform(80, 140),
            'popularity': np.random.randint(50, 95)
        }
        
        demo_data.append({
            'name': song_info['name'],
            'artist': song_info['artist'],
            **features
        })
    
    tracks_df = pd.DataFrame(demo_data)
    
    # Scale features
    scaler = StandardScaler()
    feature_array = scaler.fit_transform(tracks_df[feature_columns])
    tracks_df['features_scaled'] = [arr for arr in feature_array]
    
    # Clean up
    del feature_array
    gc.collect()
    
    print(f"Lightweight demo data created with {len(tracks_df)} tracks")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        if tracks_df is None or scaler is None:
            return jsonify({'success': False, 'error': 'Data not loaded. Please try again later.'})
        
        data = request.json
        
        # Extract audio features from request
        user_features = np.array([[
            float(data.get('danceability', 0.5)),
            float(data.get('energy', 0.5)),
            float(data.get('loudness', -10)),
            float(data.get('speechiness', 0.05)),
            float(data.get('acousticness', 0.5)),
            float(data.get('instrumentalness', 0.0)),
            float(data.get('liveness', 0.2)),
            float(data.get('valence', 0.5)),
            float(data.get('tempo', 120))
        ]])
        
        # Get recommendations
        recommendations = get_song_recommendations(user_features[0])
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def get_song_recommendations(user_features, n_recommendations=6):  # Reduced from 8 to 6
    """Get song recommendations based on audio feature similarity"""
    
    # Scale user features
    user_features_scaled = scaler.transform([user_features])[0]
    
    # Calculate cosine similarity efficiently
    similarities = []
    for idx, track_features in enumerate(tracks_df['features_scaled']):
        similarity = cosine_similarity([user_features_scaled], [track_features])[0][0]
        similarities.append((idx, similarity))
    
    # Get top recommendations
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:n_recommendations]]
    
    recommendations = []
    for idx in top_indices:
        track = tracks_df.iloc[idx]
        
        recommendations.append({
            'name': track['name'],
            'artists': track['artist'],
            'popularity': int(track['popularity']),
            'danceability': float(track['danceability']),
            'energy': float(track['energy']),
            'tempo': float(track['tempo']),
            'similarity_score': float(similarities[top_indices.index(idx)][1])
        })
    
    return recommendations

@app.route('/api/preset_styles', methods=['GET'])
def get_preset_styles():
    """Get preset music styles"""
    presets = {
        'pop': {
            'name': '🎵 Pop',
            'description': 'Catchy, upbeat songs',
            'danceability': 0.75,
            'energy': 0.8,
            'loudness': -6,
            'speechiness': 0.06,
            'acousticness': 0.2,
            'instrumentalness': 0.1,
            'liveness': 0.2,
            'valence': 0.75,
            'tempo': 120
        },
        'rock': {
            'name': '🎸 Rock',
            'description': 'Energetic guitar-driven music',
            'danceability': 0.55,
            'energy': 0.85,
            'loudness': -5,
            'speechiness': 0.05,
            'acousticness': 0.3,
            'instrumentalness': 0.4,
            'liveness': 0.4,
            'valence': 0.65,
            'tempo': 140
        },
        'chill': {
            'name': '🌙 Chill',
            'description': 'Relaxed, mellow vibes',
            'danceability': 0.5,
            'energy': 0.4,
            'loudness': -15,
            'speechiness': 0.04,
            'acousticness': 0.8,
            'instrumentalness': 0.5,
            'liveness': 0.2,
            'valence': 0.5,
            'tempo': 90
        }
    }
    
    return jsonify(presets)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if tracks_df is not None else 'unhealthy',
        'total_tracks': len(tracks_df) if tracks_df is not None else 0,
        'memory_optimized': True
    })

# Initialize data when the module loads
print("🚀 Starting optimized music recommendation app...")
load_data_and_train_model()
print("✅ App initialization completed!")

if __name__ == '__main__':
    print(f"🎵 Loaded {len(tracks_df) if tracks_df is not None else 0} tracks")
    print("🌐 Server starting on http://localhost:5000")
    app.run(debug=False, port=5000)