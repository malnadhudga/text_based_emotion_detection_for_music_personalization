from flask import Flask, render_template, request, jsonify, redirect, session
from flask_session import Session
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


# Load the model and tokenizer
def load_model_and_tokenizer():
    model_name = "emotion_classifier_8epoch"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer


# Global variables to store model and tokenizer
model, tokenizer = load_model_and_tokenizer()


def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="music_recommendation"
    )


@app.route("/")
def index():
    if not session.get("user_id"):
        return redirect("/login")
    return render_template('index.html')


@app.route("/login", methods=["POST", "GET"])
def login():
    if session.get("user_id"):
        return redirect("/")
    if request.method == "POST":
        session["name"] = request.form.get("name")
        user_id = get_user_id_by_name(session["name"])
        if not user_id:
            context = {"message": "User not found or incorrect password"}
            return render_template("login.html", **context)
        session["user_id"] = user_id[0][0]
        return redirect("/")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session["name"] = None
    session["user_id"] = None
    return redirect("/login")


def get_user_id_by_name(name):
    try:
        db = connect_db()
        cursor = db.cursor()
        query = "SELECT user_id FROM users WHERE username = %s;"
        cursor.execute(query, (name,))
        data = cursor.fetchall()
        cursor.close()
        db.close()
        return data
    except Exception as e:
        print(e)


def get_liked_songs(user_id, song_ids):
    try:
        db = connect_db()
        cursor = db.cursor()
        liked_songs_query = f"""
        SELECT song_id FROM user_likes 
        WHERE user_id = %s AND song_id IN ({','.join(map(str, song_ids))})
        """
        cursor.execute(liked_songs_query, (user_id,))
        liked_song_ids = [row[0] for row in cursor.fetchall()]
        cursor.close()
        db.close()
        return [1 if song_id in liked_song_ids else 0 for song_id in song_ids]
    except Exception as e:
        print(e)
        return []


def fetch_all_songs():
    try:
        db = connect_db()
        cursor = db.cursor()
        query = "SELECT song_id, title, artist, genre, emotion FROM songs"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        db.close()
        return pd.DataFrame(data, columns=['song_id', 'song_title', 'artist', 'genre', 'emotion'])
    except Exception as e:
        print(e)
        return pd.DataFrame()


def fetch_user_preference(user_id):
    try:
        db = connect_db()
        cursor = db.cursor()
        query = """
        SELECT DISTINCT s.artist, s.genre FROM user_likes ul
        JOIN songs s ON ul.song_id = s.song_id
        WHERE ul.user_id = %s
        """
        cursor.execute(query, (user_id,))
        data = cursor.fetchall()
        cursor.close()
        db.close()
        return pd.DataFrame(data, columns=['favorite_artists', 'preferred_genres'])
    except Exception as e:
        print(e)
        return pd.DataFrame()


def recommend_songs(user_profile, filtered_df):
    favorite_artists = user_profile['favorite_artists'].tolist()
    preferred_genres = user_profile['preferred_genres'].tolist()

    results = []
    for _, row in filtered_df.iterrows():
        artist_sim = 1 if row['artist'] in favorite_artists else 0
        genre_sim = 1 if row['genre'] in preferred_genres else 0
        similarity_score = (artist_sim + genre_sim) / 2
        results.append({
            'song_id': row['song_id'],
            'song_title': row['song_title'],
            'artist': row['artist'],
            'similarity_score': similarity_score,
        })

    return sorted(results, key=lambda x: x['similarity_score'], reverse=True)


@app.route("/", methods=['POST'])
def predict():
    para = request.form.get('para')
    inputs = tokenizer(para, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=1).item()

    emotion_map = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear'}
    predicted_emotion = emotion_map.get(predicted_class, "Unknown")

    df = fetch_all_songs()
    filtered_df = df[df['emotion'].str.lower() == predicted_emotion.lower()]

    tfidf = TfidfVectorizer(stop_words='english')
    features = tfidf.fit_transform(filtered_df['genre'] + ' ' + filtered_df['artist'])
    cosine_sim = cosine_similarity(features, features)

    user_id = session["user_id"]
    user_profile = fetch_user_preference(user_id)
    results = recommend_songs(user_profile, filtered_df)

    song_ids = [song["song_id"] for song in results[:5]]
    liked_songs = get_liked_songs(user_id, song_ids)

    return jsonify({
        'predicted_emotion': predicted_emotion,
        'results': results[:5],
        'liked_songs': liked_songs,
    })


@app.route('/toggle_like', methods=['POST'])
def toggle_like():
    user_id = session["user_id"]
    try:
        song_id = int(request.form.get('song_id'))
        action = request.form.get('action')  # 'like' or 'unlike'

        db = connect_db()
        cursor = db.cursor()

        if action == "like":
            query = "INSERT INTO user_likes (user_id, song_id) VALUES (%s, %s)"
        elif action == "unlike":
            query = "DELETE FROM user_likes WHERE user_id = %s AND song_id = %s"
        else:
            raise ValueError("Invalid action")

        cursor.execute(query, (user_id, song_id))
        db.commit()
        cursor.close()
        db.close()

        return jsonify({'success': True}), 200
    except Exception as e:
        print(e)
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

