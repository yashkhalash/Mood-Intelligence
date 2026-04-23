import os
import sqlite3
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import numpy as np

# Force CPU mode (this project is CPU-first)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress PyTorch CUDA warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
try:
    import torch
    torch.cuda.is_available = lambda: False
except ImportError:
    pass

# Import engine (backwards-compatible API, now MediaPipe + PyTorch)
from mood_detector import EmotionEngine

# Initialize Emotion Engine (auto-loads models/raf_db_emotion_model.pth if present)
engine = EmotionEngine()

app = Flask(__name__)

# Ensure required directories exist relative to the application root
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
HISTORY_FOLDER = os.path.join(app.root_path, "static", "history")
DATASET_TRAIN_DIR = "/home/ns-44/Desktop/Mood Detector/Face_Dataset/train"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HISTORY_FOLDER, exist_ok=True)

DB_PATH = os.path.join(app.root_path, "mood_history.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS history 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      timestamp TEXT,
                      mood TEXT,
                      confidence REAL,
                      image_url TEXT,
                      scores_json TEXT)''')
init_db()
init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.png', mimetype='image/png')

def get_vibe_metadata(mood):
    vibes = {
        "happy": {"label": "Radiant", "color": "#fbbf24", "glow": "rgba(251, 191, 36, 0.4)"},
        "sad": {"label": "Melancholy", "color": "#60a5fa", "glow": "rgba(96, 165, 250, 0.4)"},
        "angry": {"label": "Fiery", "color": "#f87171", "glow": "rgba(248, 113, 113, 0.4)"},
        "fear": {"label": "Tense", "color": "#a78bfa", "glow": "rgba(167, 139, 250, 0.4)"},
        "surprise": {"label": "Astonished", "color": "#2dd4bf", "glow": "rgba(45, 212, 191, 0.4)"},
        "neutral": {"label": "Stoic", "color": "#22c55e", "glow": "rgba(34, 197, 94, 0.4)"},
        "disgust": {"label": "Averse", "color": "#94a3b8", "glow": "rgba(148, 163, 184, 0.4)"}
    }
    return vibes.get(mood.lower(), {"label": "Balanced", "color": "#22c55e", "glow": "rgba(34, 197, 94, 0.4)"})

def get_mood_insights(mood):
    insights = {
        "angry": {
            "wisdom": "Patience is the best remedy for anger. Take a deep breath.",
            "recommendation": "Listen to calming lo-fi beats",
            "link": "https://open.spotify.com/playlist/37i9dQZF1DWWQRwui0Ex7X",
            "icon": "🎵"
        },
        "disgust": {
            "wisdom": "Focus on the beauty within and around you.",
            "recommendation": "Try a 5-minute mindfulness session",
            "link": "https://www.headspace.com/meditation/mindfulness",
            "icon": "🧘"
        },
        "fear": {
            "wisdom": "Courage is not the absence of fear, but the triumph over it.",
            "recommendation": "Read 'The Power of Now' summary",
            "link": "https://www.sloww.co/the-power-of-now-eckhart-tolle-summary/",
            "icon": "📖"
        },
        "happy": {
            "wisdom": "Happiness is a choice, not a result. Keep shining!",
            "recommendation": "Share your joy! Check out upbeat pop",
            "link": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlq",
            "icon": "🎸"
        },
        "sad": {
            "wisdom": "This too shall pass. Every cloud has a silver lining.",
            "recommendation": "Watch some motivational content",
            "link": "https://www.youtube.com/results?search_query=motivational+speech",
            "icon": "📽️"
        },
        "surprise": {
            "wisdom": "Life is full of surprises. Embrace the unexpected!",
            "recommendation": "Explore something new today!",
            "link": "https://www.ted.com/talks",
            "icon": "🌟"
        },
        "neutral": {
            "wisdom": "A calm mind brings inner strength and self-confidence.",
            "recommendation": "Perfect time for focused work/study",
            "link": "https://pomofocus.io/",
            "icon": "💻"
        }
    }
    return insights.get(mood.lower(), {
        "wisdom": "Stay mindful and centered.",
        "recommendation": "Take a moment for yourself",
        "link": "#",
        "icon": "✨"
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Optimization: Resize large images and ensure RGB mode for stability
        # Very important for YOLOv8 and saving as JPEG
        with Image.open(file_path) as img:
            # Handle RGBA and other non-RGB modes immediately to fix "cannot write mode RGBA as JPEG"
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            max_size = 1000
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                print(f"Resized image from {img.size} to {new_size}")
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save back as JPEG to ensure compatibility with YOLO and backend
            img.save(file_path, format='JPEG', quality=95)
            analyzed_w, analyzed_h = img.size

        # Use the new EmotionEngine for detection and prediction
        # This handles YOLOv8-face and MobileNetV2
        results = engine.process_image(file_path)

        if not results:
            return jsonify({"error": "Could not detect a clear face. Try another photo."}), 400
        all_face_results = []
        
        for idx, face in enumerate(results):
            region = face.get('region', {})
            emotion = face.get('dominant_emotion', 'Neutral')
            scores = face.get('emotion', {})
            confidence = float(face.get('face_confidence', 0))
            
            # Save cropped face for history
            thumb_filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            thumb_path = os.path.join(HISTORY_FOLDER, thumb_filename)
            
            # Save to history if high confidence
            if confidence >= 0.6:
                db_mood = emotion
                
                try:
                    with Image.open(file_path).convert('RGB') as img:
                        x_r, y_r, w_r, h_r = region['x'], region['y'], region['w'], region['h']
                        pad = 20
                        left = max(0, x_r - pad)
                        top = max(0, y_r - pad)
                        right = min(img.width, x_r + w_r + pad)
                        bottom = min(img.height, y_r + h_r + pad)
                        
                        face_thumb = img.crop((left, top, right, bottom))
                        face_thumb.thumbnail((120, 120))
                        face_thumb.save(thumb_path)
                        image_url = f"/static/history/{thumb_filename}"
                    
                        # Also save to dataset for continuous learning if confidence is very high
                        if confidence >= 0.8:
                            dataset_emotion_dir = os.path.join(DATASET_TRAIN_DIR, emotion.lower())
                            if os.path.exists(dataset_emotion_dir):
                                dataset_filename = f"collected_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                                dataset_path = os.path.join(dataset_emotion_dir, dataset_filename)
                                # Save with higher resolution for training (e.g., 96x96 as per our new model)
                                training_face = img.crop((left, top, right, bottom))
                                training_face = training_face.resize((96, 96), Image.Resampling.LANCZOS)
                                training_face.save(dataset_path)
                                print(f"Saved new training sample to {dataset_path}")
                except Exception as e:
                    print(f"Thumb save failed: {e}")
                    image_url = None

                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO history (mood, timestamp, image_url, scores_json) VALUES (?, ?, ?, ?)",
                        (db_mood, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), image_url, json.dumps(scores))
                    )
            else:
                image_url = None 
                print(f"Skipping history save: Low confidence face ({confidence})")

            insights = get_mood_insights(emotion)
            vibe = get_vibe_metadata(emotion)
            
            all_face_results.append({
                "face_index": idx + 1,
                "mood": emotion,
                "vibe": vibe['label'],
                "theme_color": vibe['color'],
                "theme_glow": vibe['glow'],
                "scores": scores,
                "confidence": confidence,
                "region": region,
                "wisdom": insights['wisdom'],
                "recommendation": insights['recommendation'],
                "link": insights['link'],
                "rec_icon": insights['icon'],
                "image_url": image_url
            })
            
        # Get model metadata if exists
        metadata = {}
        metadata_path = os.path.join(app.root_path, "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except:
                pass

        return jsonify({
            "faces": all_face_results,
            "face_count": len(all_face_results),
            "analyzed_image": {"width": analyzed_w, "height": analyzed_h},
            "model_metadata": metadata,
        })

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/history')
def get_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT * FROM history ORDER BY id DESC LIMIT 20")
        rows = cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get('scores_json'):
                d['scores'] = json.loads(d['scores_json'])
            result.append(d)
        return jsonify(result)

@app.route('/stats')
def get_stats():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT mood, COUNT(*) as count FROM history GROUP BY mood")
        data = cursor.fetchall()
        return jsonify({row[0]: row[1] for row in data})

@app.route('/history/delete/<int:item_id>', methods=['DELETE'])
def delete_history(item_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT image_url FROM history WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            if row and row['image_url']:
                # Delete physical file
                file_path = row['image_url'].lstrip('/')
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            conn.execute("DELETE FROM history WHERE id = ?", (item_id,))
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Disable reloader for stability during model loads
    # Enable threaded for better handling of concurrent requests/timeouts
    app.run(debug=False, host='0.0.0.0', use_reloader=False, threaded=True)