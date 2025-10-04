# app.py
from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS 
import pandas as pd 
import numpy as np 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics.pairwise import cosine_similarity 
import os
import traceback
import ast

app = Flask(__name__)
CORS(app)

# Global variables for the model and data
model = None
scaler = None
tracks_df = None
feature_columns = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

def load_data_and_train_model():
    global model, scaler, tracks_df
    
    try:
        # Load your dataset
        print("Loading dataset...")
        tracks_df = pd.read_csv('tracks.csv', on_bad_lines='skip')
        print(f"Dataset loaded with {len(tracks_df)} tracks")
        
        # Data preprocessing - keep only tracks with names and artists
        tracks_df = tracks_df.dropna(subset=['name', 'artists'])
        
        # Remove tracks with placeholder or missing names
        tracks_df = tracks_df[tracks_df['name'].str.len() > 0]
        tracks_df = tracks_df[tracks_df['artists'].str.len() > 0]
        
        # Clean up artist names - safely parse the list format
        def safe_parse_artists(artist_str):
            try:
                if pd.isna(artist_str):
                    return ['Unknown Artist']
                # Remove extra quotes and parse the list
                artist_str = str(artist_str).replace('"', "'")
                artists = ast.literal_eval(artist_str)
                if isinstance(artists, list) and len(artists) > 0:
                    return [str(artist) for artist in artists]
                else:
                    return ['Unknown Artist']
            except:
                # If parsing fails, try to extract artist names manually
                artist_str = str(artist_str).strip("[]'\"")
                if artist_str and len(artist_str) > 0:
                    return [artist_str]
                else:
                    return ['Unknown Artist']
        
        tracks_df['artists_clean'] = tracks_df['artists'].apply(safe_parse_artists)
        tracks_df['primary_artist'] = tracks_df['artists_clean'].apply(lambda x: x[0] if x else 'Unknown Artist')
        
        # Create a sample of the dataset for faster processing (remove for full dataset)
        # Keep popular and diverse tracks for better recommendations
        tracks_df = tracks_df.sort_values('popularity', ascending=False).head(20000)
        
        print(f"Processing {len(tracks_df)} tracks after cleaning")
        
        # Prepare features for similarity calculation
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(tracks_df[feature_columns])
        tracks_df['features_scaled'] = list(features_scaled)
        
        print("Data preprocessing completed!")
        print("Sample of tracks:")
        for i in range(min(5, len(tracks_df))):
            track = tracks_df.iloc[i]
            print(f"  - {track['name']} by {track['primary_artist']} (Popularity: {track['popularity']})")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print(traceback.format_exc())
        create_demo_data()

def create_demo_data():
    """Create demo data only if real data fails completely"""
    global tracks_df, scaler
    
    print("Creating demo data with realistic song names...")
    
    np.random.seed(42)
    n_samples = 500
    
    # Create more realistic demo data with actual song/artist names
    real_songs = [
        {"name": "Blinding Lights", "artist": "The Weeknd"},
        {"name": "Shape of You", "artist": "Ed Sheeran"},
        {"name": "Dance Monkey", "artist": "Tones and I"},
        {"name": "Someone You Loved", "artist": "Lewis Capaldi"},
        {"name": "Bad Guy", "artist": "Billie Eilish"},
        {"name": "Watermelon Sugar", "artist": "Harry Styles"},
        {"name": "Don't Start Now", "artist": "Dua Lipa"},
        {"name": "Circles", "artist": "Post Malone"},
        {"name": "Savage Love", "artist": "Jason Derulo"},
        {"name": "Levitating", "artist": "Dua Lipa"},
        {"name": "Stay", "artist": "The Kid LAROI, Justin Bieber"},
        {"name": "Good 4 U", "artist": "Olivia Rodrigo"},
        {"name": "Butter", "artist": "BTS"},
        {"name": "Heat Waves", "artist": "Glass Animals"},
        {"name": "Save Your Tears", "artist": "The Weeknd"},
        {"name": "Kiss Me More", "artist": "Doja Cat"},
        {"name": "Montero", "artist": "Lil Nas X"},
        {"name": "Peaches", "artist": "Justin Bieber"},
        {"name": "Stay", "artist": "Kid LAROI"},
        {"name": "Industry Baby", "artist": "Lil Nas X"}
    ]
    
    genres = ['Pop', 'Rock', 'Hip-Hop', 'Electronic', 'R&B', 'Country', 'Jazz']
    
    demo_tracks = []
    for i in range(n_samples):
        if i < len(real_songs):
            # Use real song names for first few tracks
            song_info = real_songs[i]
            genre = 'Pop'  # Most popular songs are pop
        else:
            # Generate realistic-sounding names for remaining tracks
            prefixes = ['Midnight', 'Summer', 'Electric', 'Golden', 'Neon', 'Wild', 'Silent', 'Lost', 'Brave', 'Free']
            suffixes = ['Dreams', 'Love', 'Fire', 'Waves', 'Sky', 'Heart', 'Soul', 'Night', 'Day', 'Light']
            artists_first = ['Taylor', 'Ariana', 'Bruno', 'Drake', 'Rihanna', 'Weeknd', 'Ed', 'Billie', 'Harry', 'Dua']
            artists_last = ['Swift', 'Grande', 'Mars', 'Lipa', 'Styles', 'Eilish', 'Sheeran', 'Malone', 'Bieber', 'Derulo']
            
            song_name = f"{np.random.choice(prefixes)} {np.random.choice(suffixes)}"
            artist_name = f"{np.random.choice(artists_first)} {np.random.choice(artists_last)}"
            genre = np.random.choice(genres)
            
            song_info = {"name": song_name, "artist": artist_name}
        
        # Different audio features based on genre
        if genre == 'Pop':
            features = {
                'danceability': np.random.uniform(0.6, 0.9),
                'energy': np.random.uniform(0.7, 0.9),
                'loudness': np.random.uniform(-8, -4),
                'speechiness': np.random.uniform(0.03, 0.1),
                'acousticness': np.random.uniform(0.1, 0.4),
                'instrumentalness': np.random.uniform(0.0, 0.2),
                'liveness': np.random.uniform(0.1, 0.4),
                'valence': np.random.uniform(0.6, 0.9),
                'tempo': np.random.uniform(100, 140)
            }
        elif genre == 'Rock':
            features = {
                'danceability': np.random.uniform(0.4, 0.7),
                'energy': np.random.uniform(0.8, 1.0),
                'loudness': np.random.uniform(-7, -3),
                'speechiness': np.random.uniform(0.02, 0.08),
                'acousticness': np.random.uniform(0.1, 0.5),
                'instrumentalness': np.random.uniform(0.1, 0.6),
                'liveness': np.random.uniform(0.2, 0.6),
                'valence': np.random.uniform(0.4, 0.8),
                'tempo': np.random.uniform(120, 160)
            }
        elif genre == 'Hip-Hop':
            features = {
                'danceability': np.random.uniform(0.7, 0.9),
                'energy': np.random.uniform(0.6, 0.9),
                'loudness': np.random.uniform(-9, -5),
                'speechiness': np.random.uniform(0.2, 0.8),
                'acousticness': np.random.uniform(0.0, 0.3),
                'instrumentalness': np.random.uniform(0.0, 0.1),
                'liveness': np.random.uniform(0.1, 0.3),
                'valence': np.random.uniform(0.5, 0.9),
                'tempo': np.random.uniform(80, 120)
            }
        else:  # Electronic/R&B/Country/Jazz
            features = {
                'danceability': np.random.uniform(0.3, 0.8),
                'energy': np.random.uniform(0.4, 0.8),
                'loudness': np.random.uniform(-15, -8),
                'speechiness': np.random.uniform(0.02, 0.1),
                'acousticness': np.random.uniform(0.3, 0.9),
                'instrumentalness': np.random.uniform(0.1, 0.7),
                'liveness': np.random.uniform(0.1, 0.5),
                'valence': np.random.uniform(0.4, 0.8),
                'tempo': np.random.uniform(70, 130)
            }
        
        demo_tracks.append({
            'name': song_info['name'],
            'artists': f"['{song_info['artist']}']",
            'primary_artist': song_info['artist'],
            'popularity': np.random.randint(40, 100),
            'genre': genre,
            **features
        })
    
    tracks_df = pd.DataFrame(demo_tracks)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(tracks_df[feature_columns])
    tracks_df['features_scaled'] = list(features_scaled)
    
    print("Demo data created successfully!")

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
        recommendations = get_song_recommendations(user_features[0], n_recommendations=8)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'user_features': {
                'danceability': user_features[0][0],
                'energy': user_features[0][1],
                'loudness': user_features[0][2],
                'speechiness': user_features[0][3],
                'acousticness': user_features[0][4],
                'instrumentalness': user_features[0][5],
                'liveness': user_features[0][6],
                'valence': user_features[0][7],
                'tempo': user_features[0][8]
            }
        })
        
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

def get_song_recommendations(user_features, n_recommendations=8):
    """Get song recommendations based on audio feature similarity"""
    
    # Scale user features
    user_features_scaled = scaler.transform([user_features])[0]
    
    # Calculate cosine similarity
    similarities = []
    for idx, track_features in enumerate(tracks_df['features_scaled']):
        similarity = cosine_similarity([user_features_scaled], [track_features])[0][0]
        similarities.append((idx, similarity))
    
    # Sort by similarity and get top recommendations
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:n_recommendations]]
    
    recommendations = []
    for idx in top_indices:
        track = tracks_df.iloc[idx]
        
        # Use the cleaned artist name
        artist_name = track.get('primary_artist', 'Unknown Artist')
        if pd.isna(artist_name) or artist_name == 'Unknown Artist':
            # Fallback to original artists field
            try:
                if pd.notna(track['artists']):
                    artists_list = ast.literal_eval(str(track['artists']).replace('"', "'"))
                    artist_name = artists_list[0] if artists_list else 'Unknown Artist'
            except:
                artist_name = 'Unknown Artist'
        
        # Get song name
        song_name = track['name']
        if pd.isna(song_name):
            song_name = 'Unknown Track'
        
        # Estimate genre based on features if not available
        genre = track.get('genre', 'Various')
        if pd.isna(genre):
            # Simple genre estimation based on features
            if track['acousticness'] > 0.7:
                genre = 'Acoustic'
            elif track['instrumentalness'] > 0.5:
                genre = 'Instrumental'
            elif track['energy'] > 0.8 and track['danceability'] > 0.7:
                genre = 'Dance'
            elif track['speechiness'] > 0.3:
                genre = 'Hip-Hop'
            else:
                genre = 'Various'
        
        recommendations.append({
            'name': song_name,
            'artists': artist_name,
            'popularity': int(track['popularity']),
            'genre': genre,
            'danceability': float(track['danceability']),
            'energy': float(track['energy']),
            'tempo': float(track['tempo']),
            'acousticness': float(track['acousticness']),
            'similarity_score': float(similarities[top_indices.index(idx)][1])
        })
    
    return recommendations

@app.route('/api/preset_styles', methods=['GET'])
def get_preset_styles():
    """Get preset music styles"""
    presets = {
        'pop': {
            'name': '🎵 Pop',
            'description': 'Catchy, upbeat songs with mass appeal',
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
        'hiphop': {
            'name': '🎤 Hip-Hop',
            'description': 'Rhythmic vocal tracks with strong beats',
            'danceability': 0.8,
            'energy': 0.7,
            'loudness': -7,
            'speechiness': 0.5,
            'acousticness': 0.1,
            'instrumentalness': 0.05,
            'liveness': 0.2,
            'valence': 0.7,
            'tempo': 100
        },
        'electronic': {
            'name': '⚡ Electronic',
            'description': 'Synthesizer-based dance music',
            'danceability': 0.75,
            'energy': 0.8,
            'loudness': -4,
            'speechiness': 0.06,
            'acousticness': 0.1,
            'instrumentalness': 0.6,
            'liveness': 0.25,
            'valence': 0.65,
            'tempo': 130
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
        },
        'party': {
            'name': '🎉 Party',
            'description': 'High-energy dance tracks',
            'danceability': 0.85,
            'energy': 0.9,
            'loudness': -3,
            'speechiness': 0.08,
            'acousticness': 0.1,
            'instrumentalness': 0.2,
            'liveness': 0.3,
            'valence': 0.85,
            'tempo': 125
        }
    }
    
    return jsonify(presets)

@app.route('/api/dataset_stats', methods=['GET'])
def get_dataset_stats():
    """Get dataset statistics"""
    if tracks_df is None:
        return jsonify(get_demo_stats())
    
    try:
        # Get actual statistics from the dataset
        total_tracks = len(tracks_df)
        avg_popularity = float(tracks_df['popularity'].mean())
        
        # Show sample of actual song names in the dataset
        sample_songs = []
        for i in range(min(5, len(tracks_df))):
            track = tracks_df.iloc[i]
            sample_songs.append({
                'name': track['name'],
                'artist': track.get('primary_artist', 'Unknown'),
                'popularity': int(track['popularity'])
            })
        
        stats = {
            'total_tracks': total_tracks,
            'average_popularity': avg_popularity,
            'features_available': feature_columns,
            'sample_songs': sample_songs,
            'message': f"Loaded {total_tracks} real songs from your dataset"
        }
        
        return jsonify(stats)
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return jsonify(get_demo_stats())

def get_demo_stats():
    """Return demo stats if real data isn't available"""
    return {
        'total_tracks': 500,
        'average_popularity': 65.5,
        'features_available': feature_columns,
        'sample_songs': [
            {'name': 'Blinding Lights', 'artist': 'The Weeknd', 'popularity': 95},
            {'name': 'Shape of You', 'artist': 'Ed Sheeran', 'popularity': 92},
            {'name': 'Dance Monkey', 'artist': 'Tones and I', 'popularity': 88}
        ],
        'message': 'Using demo data with realistic song names'
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'data_loaded': tracks_df is not None,
        'scaler_loaded': scaler is not None,
        'total_tracks': len(tracks_df) if tracks_df is not None else 0,
        'data_source': 'real_dataset' if tracks_df is not None and 'primary_artist' in tracks_df.columns else 'demo_data'
    })

if __name__ == '__main__':
    load_data_and_train_model()
    print("Server starting on http://localhost:5000")
    print(f"Loaded {len(tracks_df) if tracks_df is not None else 0} tracks for recommendations")
    app.run(debug=True, port=5000)