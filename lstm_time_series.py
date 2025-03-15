import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from pytube import YouTube
from pytrends.request import TrendReq
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

file_path = "moststreamed2024.csv"
encoding = detect_encoding(file_path)
df = pd.read_csv(file_path, encoding=encoding)

features = [
    'Spotify Streams', 'YouTube Views', 'TikTok Likes', 'Spotify Playlist Count',
    'Spotify Playlist Reach', 'Spotify Popularity', 'Release Date', 'Artist',
    'Explicit Track', 'TikTok Posts', 'TikTok Views', 'YouTube Likes', 'YouTube Playlist Reach'
]
target = 'Track Score'

df = df.dropna(subset=features + [target])
df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df = df.dropna(subset=['Release Date'])
df['Release Year'] = df['Release Date'].dt.year
df['Release Month'] = df['Release Date'].dt.month
df['Release Day'] = df['Release Date'].dt.day

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(df[['Artist']])
df_encoded = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Artist']))
df = pd.concat([df.reset_index(drop=True), df_encoded], axis=1)
df = df.drop(columns=['Artist', 'Release Date'])

numerical_features = [
    'Spotify Streams', 'YouTube Views', 'TikTok Likes', 'Spotify Playlist Count',
    'Spotify Playlist Reach', 'Spotify Popularity', 'TikTok Posts', 'TikTok Views',
    'YouTube Likes', 'YouTube Playlist Reach', 'Release Year', 'Release Month', 'Release Day'
]

for col in numerical_features:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

def create_sequences(data, seq_length):
    Xs, ys = [], []
    for i in range(len(data) - seq_length):
        X = data.iloc[i:(i + seq_length)][numerical_features + list(df_encoded.columns)].values
        y = data.iloc[i + seq_length][target]
        Xs.append(X)
        ys.append(y)
    return np.array(Xs), np.array(ys)

seq_length = 10
X, y = create_sequences(df, seq_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss}')

predictions = model.predict(X_test)
for i in range(10):
    print(f'Predicted: {predictions[i][0]}, Actual: {y_test[i]}')

model.save('song_hit_trend_lstm_enhanced.h5')

def fetch_youtube_engagement(video_url):
    try:
        yt = YouTube(video_url)
        return yt.views, yt.likes, len(yt.comments)
    except Exception as e:
        print(f"Error fetching YouTube data: {e}")
        return 0, 0, 0

def fetch_google_trends(song_name, artist_name):
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = [f"{song_name} {artist_name}"]
        pytrends.build_payload(kw_list, cat=0, timeframe='today 12-m', geo='', gprop='')
        interest_over_time_df = pytrends.interest_over_time()
        return interest_over_time_df[kw_list[0]].sum() if kw_list[0] in interest_over_time_df else 0
    except Exception as e:
        print(f"Error fetching Google Trends data: {e}")
        return 0

if 'YouTube URL' in df.columns:
    df['YouTube Views'], df['YouTube Likes'], df['YouTube Comments'] = zip(*df['YouTube URL'].apply(fetch_youtube_engagement))

df['Google Trends Interest'] = df.apply(lambda row: fetch_google_trends(row['Track'], row['Artist']) if 'Track' in df.columns and 'Artist' in df.columns else 0, axis=1)

df.to_csv('moststreamed2024_with_engagement.csv', index=False)

from tensorflow.keras.models import load_model
model = load_model('song_hit_trend_lstm_enhanced.h5')

new_data = df.tail(seq_length)[numerical_features + list(df_encoded.columns)].values
new_data = np.array([new_data])

predicted_score = model.predict(new_data)
print(f'Predicted Track Score: {predicted_score[0][0]}')
