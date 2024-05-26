import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from datetime import datetime
from jinja2 import Environment

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Google Generative AI
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# Creating Flask app and linking it to Mongo_DB and creating login credentials
app = Flask(__name__)
app.secret_key = "your_secret_key"

app.config["MONGO_URI"] = "mongodb://localhost:27017/CFRS_db"
mongo = PyMongo(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, username, password, id=None):
        self.username = username
        self.password = password
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    user_data = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user_data:
        return User(username=user_data['username'], password=user_data['password'], id=str(user_data['_id']))
    return None

# Jinja environment with the `str` function
env = Environment()
env.filters['str'] = str

# Define the zip_lists function
def zip_lists(list1, list2):
    return zip(list1, list2)

# Register the zip_lists function as a custom filter named 'zip'
app.jinja_env.filters['zip'] = zip_lists

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = mongo.db.users
        login_user_data = users.find_one({'username': request.form['username']})

        if login_user_data and check_password_hash(login_user_data['password'], request.form['password']):
            user = User(username=login_user_data['username'], password=login_user_data['password'], id=str(login_user_data['_id']))
            login_user(user)
            return redirect(url_for('index'))

        return 'Invalid username/password combination'

    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'username': request.form['username']})

        if existing_user is None:
            hashpass = generate_password_hash(request.form['password'])
            user_id = users.insert_one({'username': request.form['username'], 'password': hashpass}).inserted_id
            user = User(username=request.form['username'], password=hashpass, id=str(user_id))
            login_user(user)
            return redirect(url_for('index'))

        return 'That username already exists!'

    return redirect(url_for('index'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Route for rendering the chat page
@app.route('/chat')
def chat_page():
    return render_template('chat.html')

# Route for handling chat messages
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    question = data['message']
    response = chat.send_message(question)
    return jsonify(reply=response.text)

# Crop recommendation model
# =========================================================================
@app.route('/crop')
def crop_recommendation():
    return render_template("crop.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = request.form.get('Nitrogen')
    P = request.form.get('Phosporus')
    K = request.form.get('Potassium')
    temp = request.form.get('Temperature')
    humidity = request.form.get('Humidity')
    ph = request.form.get('Ph')
    rainfall = request.form.get('Rainfall')

    model, accuracy = recommendation_model(df)

    result = ["", "", ""]
    prediction, res = recommendation(N, P, K, temp, humidity, ph, rainfall, model)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }
    for i in range(3):
        if prediction[i] in crop_dict:
            crop = crop_dict[prediction[i]]
            result[i] = "{} is the best crop to be cultivated right there".format(crop)
        else:
            result[i] = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    store_prediction(current_user.id, res, temp, humidity, ph, rainfall)
    return render_template('crop.html', result=result, accuracy=accuracy)

# history
@app.route('/history')
@login_required
def history():
    crop_predictions = fetch_prediction_history(current_user.id)
    return render_template('history.html', crop_predictions=crop_predictions)

# Fertilizer recommendation model
# ============================================================================================================
@app.route('/fertilizer')
def fertilizer_recommendation():
    return render_template("fertilizer.html")

@app.route("/fertilizer_predict", methods=['POST'])
def fertilizer_predict():
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    moisture = float(request.form['Moisture'])
    soil_type = request.form['Soil_Type']
    crop_type = request.form['Crop_Type']
    nitro = float(request.form['Nitrogen'])
    pot = float(request.form['Potassium'])
    phosp = float(request.form['Phosphorous'])

    # Encode categorical variables
    soil_type_encoded = encode_soil.transform([soil_type])[0]
    crop_type_encoded = encode_crop.transform([crop_type])[0]

    # Prepare input data
    input_data = pd.DataFrame([[temp, humidity, moisture, soil_type_encoded, crop_type_encoded, nitro, pot, phosp]], 
                              columns=['Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous'])

    # Predict fertilizer
    fertilizer_prediction = fertilizer_model.predict(input_data)
    predicted_fertilizer = encode_ferti.inverse_transform(fertilizer_prediction)

    return render_template('fertilizer.html', result=predicted_fertilizer[0])

# Fertilizer recommendation data preparation
fertilizer_data = pd.read_csv("Fertilizer Prediction.csv")
fertilizer_data.rename(columns={'Humidity ': 'Humidity', 'Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer'}, inplace=True)

# Ensure that the 'Temperature' column is correctly named
fertilizer_data.rename(columns={'Temparature': 'Temperature'}, inplace=True)

encode_soil = LabelEncoder()
fertilizer_data.Soil_Type = encode_soil.fit_transform(fertilizer_data.Soil_Type)

encode_crop = LabelEncoder()
fertilizer_data.Crop_Type = encode_crop.fit_transform(fertilizer_data.Crop_Type)

encode_ferti = LabelEncoder()
fertilizer_data.Fertilizer = encode_ferti.fit_transform(fertilizer_data.Fertilizer)

x_fertilizer = fertilizer_data.drop('Fertilizer', axis=1)
y_fertilizer = fertilizer_data.Fertilizer

x_train_fertilizer, x_test_fertilizer, y_train_fertilizer, y_test_fertilizer = train_test_split(x_fertilizer, y_fertilizer, test_size=0.2, random_state=1)

fertilizer_model = RandomForestClassifier()
fertilizer_model.fit(x_train_fertilizer, y_train_fertilizer)

# Crop recommendation data preparation
crop = pd.read_csv("Crop_recommendation.csv")
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}
df = crop.copy()  # Make a copy of the DataFrame to avoid modifying the original data
df['crop_num'] = df['label'].map(crop_dict)
df.drop('label', axis=1, inplace=True)

# Crop recommendation functions
def recommendation_model(df):
    x = df.drop('crop_num', axis=1)
    y = df['crop_num']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    ms = MinMaxScaler()
    ms.fit(x_train)
    x_train = ms.transform(x_train)
    x_test = ms.transform(x_test)

    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)

    ypred = rfc.predict(x_test)
    accuracy = accuracy_score(y_test, ypred)
    
    return rfc, accuracy

def prediction_model(features, model):
    probabilities = model.predict_proba(features.reshape(1, -1))
    top_three = [sorted(zip(model.classes_, prob), key=lambda x: x[1], reverse=True)[:3] for prob in probabilities]
    
    return top_three

def recommendation(N, P, K, temperature, humidity, ph, rainfall, model):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    result = []
    res = []
    top_three = prediction_model(features, model)
    for i in top_three[0]:
        result.append(i[0])
        crop_name = get_crop_name(i[0])
        if crop_name != "Unknown":
            res.append({'crop_name': crop_name, 'N': N, 'P': P, 'K': K})
    
    return result, res

# Utility functions
def store_prediction(user_id, crop_prediction, temperature, humidity, ph, rainfall):
    current_time = datetime.now()
    mongo.db.crop_prediction_history.insert_one({
        'user_id': ObjectId(user_id),
        'crop_prediction': crop_prediction,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall,
        'timestamp': current_time
    })

def fetch_prediction_history(user_id):
    crop_predictions = mongo.db.crop_prediction_history.find({"user_id": ObjectId(user_id)})
    history = []
    for prediction in crop_predictions:
        crop_data = {
            'crop_predictions': prediction.get('crop_prediction', []),
            'temperature': prediction.get('temperature', 'N/A'),
            'humidity': prediction.get('humidity', 'N/A'),
            'ph': prediction.get('ph', 'N/A'),
            'rainfall': prediction.get('rainfall', 'N/A'),
            'timestamp': prediction['timestamp']
        }
        history.append(crop_data)
    return history

def get_crop_name(crop_num):
    for crop, num in crop_dict.items():
        if num == crop_num:
            return crop
    return "Unknown"

# Main function to run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
