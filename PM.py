from flask import Flask, jsonify
from flask_cors import CORS
import requests
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import backend as K
import threading
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
CORS(app)  # Allow all origins

# Initialize global variables
data = {}
models = {}
predictions = {}
actions = []
lock = threading.Lock()

# Function to fetch data from ThingSpeak
def fetch_data_from_thingspeak(channel_id, field_num):
    url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{field_num}.json?results=5000"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        values = [float(entry[f'field{field_num}']) if entry.get(f'field{field_num}') is not None else None for entry in data['feeds']]
        return np.array(values, dtype=np.float32)
    else:
        print("Failed to fetch data from ThingSpeak.")
        return None

# Function to replace null or zero values with the mean of neighboring values
def replace_invalid_values(data):
    valid_indices = np.where((data != 0) & ~np.isnan(data))[0]
    if len(valid_indices) == 0:
        print("No valid data points available.")
        return data
    
    for i in range(len(data)):
        if data[i] == 0 or np.isnan(data[i]):
            if i > 0 and i < len(data) - 1:
                prev_valid = valid_indices[valid_indices < i].max() if len(valid_indices[valid_indices < i]) > 0 else None
                next_valid = valid_indices[valid_indices > i].min() if len(valid_indices[valid_indices > i]) > 0 else None
                if prev_valid is not None and next_valid is not None:
                    data[i] = (data[prev_valid] + data[next_valid]) / 2
                elif prev_valid is not None:
                    data[i] = data[prev_valid]
                elif next_valid is not None:
                    data[i] = data[next_valid]
            elif i == 0:
                next_valid = valid_indices[valid_indices > i].min() if len(valid_indices[valid_indices > i]) > 0 else None
                if next_valid is not None:
                    data[i] = data[next_valid]
            elif i == len(data) - 1:
                prev_valid = valid_indices[valid_indices < i].max() if len(valid_indices[valid_indices < i]) > 0 else None
                if prev_valid is not None:
                    data[i] = data[prev_valid]
    return data

# Function to define and create LSTM models for each parameter
def create_models(train_data):
    # Define custom activation functions to constrain output ranges
    def custom_activation_ph(x):
        return K.clip(x, 6.5, 9.0)

    def custom_activation_do(x):
        return K.clip(x, 5, 20)

    def custom_activation_temp(x):
        return K.clip(x, 25, 32)

    def custom_activation_am(x):
        return K.clip(x, 0, 1)

    models = {}
    for key in train_data.keys():
        model = Sequential()
        model.add(LSTM(10, input_shape=(train_data[key].shape[1], 1)))
        if key == 'ph':
            model.add(Dense(train_data[key].shape[1], activation=custom_activation_ph))
        elif key == 'do':
            model.add(Dense(train_data[key].shape[1], activation=custom_activation_do))
        elif key == 'temp':
            model.add(Dense(train_data[key].shape[1], activation=custom_activation_temp))
        elif key == 'am':
            model.add(Dense(train_data[key].shape[1], activation=custom_activation_am))
        model.add(Dense(train_data[key].shape[1], activation='linear'))
        model.compile(optimizer='adam', loss='mse')
        models[key] = model
    return models

# Function to train models on the latest data
def train_models(models, data):
    train_data = {key: values[:-4500].reshape((1, -1, 1)) for key, values in data.items()}
    for key in train_data.keys():
        models[key].fit(train_data[key], train_data[key], epochs=100, verbose=0)
    return models

# Function to make predictions
def make_predictions(models, data):
    predictions = {}
    for key in data.keys():
        predict_data = data[key][-50:]
        predictions[key] = models[key].predict(predict_data.reshape((1, -1, 1))).flatten()
    return predictions

# Function to determine actions based on predictions
def determine_actions(predictions):
    ranges = {
        'ph': [6.5, 9.0],
        'do': [3, 20],
        'temp': [20, 32],
        'am': [0, 1]
    }
    actions = []

    # Check pH
    if any(pred < ranges['ph'][0] for pred in predictions['ph']):
        actions.append('Add lime to increase pH')
    elif any(pred > ranges['ph'][1] for pred in predictions['ph']):
        actions.append('Add acid to decrease pH')

    # Check dissolved oxygen (DO)
    if any(pred < ranges['do'][0] for pred in predictions['do']):
        actions.append('Increase aeration or reduce stocking density')
    elif any(pred > ranges['do'][1] for pred in predictions['do']):
        actions.append('Reduce aeration to avoid supersaturation')

    # Check temperature
    if any(pred < ranges['temp'][0] for pred in predictions['temp']):
        actions.append('Use heaters to increase temperature')
    elif any(pred > ranges['temp'][1] for pred in predictions['temp']):
        actions.append('Use coolers or increase shading to reduce temperature')

    # Check ammonia
    if any(pred < ranges['am'][0] for pred in predictions['am']):
        actions.append('Ensure proper biofiltration')
    elif any(pred > ranges['am'][1] for pred in predictions['am']):
        actions.append('Increase water exchange and reduce feeding')

    # General actions based on any parameter being out of range
    if (any(pred < ranges['ph'][0] or pred > ranges['ph'][1] for pred in predictions['ph']) or
        any(pred < ranges['do'][0] or pred > ranges['do'][1] for pred in predictions['do']) or
        any(pred < ranges['temp'][0] or pred > ranges['temp'][1] for pred in predictions['temp']) or
        any(pred < ranges['am'][0] or pred > ranges['am'][1] for pred in predictions['am'])):
        actions.append('Water exchange and Aeration Adjustment')

    # Specific actions for combined parameters being out of range
    if (any(pred < ranges['do'][0] or pred > ranges['do'][1] for pred in predictions['do']) and
        any(pred < ranges['temp'][0] or pred > ranges['temp'][1] for pred in predictions['temp'])):
        actions.append('Shade the system and reduce feeding')

    return actions

# Background task to fetch data, retrain models, and make predictions
def background_task():
    global data, models, predictions, actions
    while True:
        new_data = {}
        for key, field_num in field_nums.items():
            raw_data = fetch_data_from_thingspeak(channel_id, field_num)
            if raw_data is not None:
                new_data[key] = replace_invalid_values(raw_data)
        with lock:
            data = new_data
            models = train_models(models, data)
            predictions = make_predictions(models, data)
            actions = determine_actions(predictions)
        time.sleep(300)

# Initialize data and models
channel_id = 2210102
field_nums = {'ph': 3, 'do': 1, 'temp': 2, 'am': 4}

for key, field_num in field_nums.items():
    raw_data = fetch_data_from_thingspeak(channel_id, field_num)
    if raw_data is not None:
        data[key] = replace_invalid_values(raw_data)

train_data = {key: values[:-4500].reshape((1, -1, 1)) for key, values in data.items()}
models = create_models(train_data)
models = train_models(models, data)
predictions = make_predictions(models, data)
actions = determine_actions(predictions)

# Start the background task
threading.Thread(target=background_task, daemon=True).start()

# Route to get prediction data
@app.route('/prediction', methods=['GET'])
def get_prediction():
    with lock:
        prediction_data = {
            'ph': predictions['ph'].tolist(),
            'dissolvedOxygen': predictions['do'].tolist(),
            'temp': predictions['temp'].tolist(),
            'am': predictions['am'].tolist(),
            'actions': actions
        }
    return jsonify(prediction_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
