from flask import Flask, jsonify
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Connect to MongoDB
client = MongoClient('mongodb+srv://vinay:somethingforcloud@cloud-project.pwk1tsn.mongodb.net/')
db = client['greenhouse']
training_collection = db['values']
sensor_collection = db['sensordatas']
prediction_collection = db['sunlightdatas']

def train_model():
    # Load the dataset from MongoDB
    df = pd.DataFrame(list(training_collection.find()))
    X = df[['temperature', 'humidity', 'soil_moisture']]
    y = df['sunlight_reduction']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Linear Regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set for evaluation
    predictions = model.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return model

# Train the model initially
trained_model = train_model()

# Initialize global variables
prev_timestamp = None

@app.route('/')
def hello_world():
    return 'greenhouse sunlight allowance calculator'

@app.route('/predict_sunlight_reduction', methods=['GET'])
def predict_sunlight_reduction():
    global prev_timestamp

    try:
        # Get the latest data from the MongoDB database
        latest_data = sensor_collection.find_one({}, sort=[('_id', -1)])
        
        # Ensure that the required fields are present
        if not all(field in latest_data for field in ['temperature', 'humidity', 'soil_moisture']):
            return jsonify({'error': 'Temperature, humidity, and soil_moisture fields are required in the database.'})

        # Extract the latest timestamp
        latest_timestamp = latest_data['timestamp']

        # Check if there are new sensor values
        if latest_timestamp != prev_timestamp:
            prev_timestamp = latest_timestamp

            latest_temperature = latest_data['temperature']
            latest_humidity = latest_data['humidity']
            latest_soil_moisture = latest_data['soil_moisture']

            # Make a prediction for the new data
            new_data_df = pd.DataFrame({'temperature': [latest_temperature], 'humidity': [latest_humidity], 'soil_moisture': [latest_soil_moisture]})
            predicted_sunlight_reduction = trained_model.predict(new_data_df)[0]

            # Log the prediction to a different collection
            timestamp = datetime.now()
            prediction_collection.insert_one({
                'timestamp': timestamp,
                'predicted_sunlight_reduction': predicted_sunlight_reduction
            })

            return jsonify({'predicted_sunlight_reduction': predicted_sunlight_reduction})
        else:
            # No changes in sensor values, return a message indicating no calculation needed
            return jsonify({'message': 'No changes in sensor values. Sunlight reduction calculation not needed.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5004)
