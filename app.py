import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request, redirect, url_for,jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId




# Load environment variables
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# Configure Google Generative AI
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])


# Creating Flask app and linking it to Mongo_DB and creating login credentials 
# ======================================================================
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
@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']
    
    model, accuracy = recommendation_model(df)
    
    result=["","",""]
    prediction = recommendation(N, P, K, temp, humidity, ph, rainfall,model)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    for i in range(3):
        if prediction[i] in crop_dict:
            crop = crop_dict[prediction[i]]
            result[i] = "{} is the best crop to be cultivated right there".format(crop)
        else:
            result[i] = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('crop.html',result = result,accuracy=accuracy)


# Recoomendation Model
# ============================================================================================================
crop= pd.read_csv("Crop_recommendation.csv")

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
    # print("Accuracy : ",accuracy)
    
    return rfc, accuracy


def prediction_model( features,model):
    probabilities = model.predict_proba(features.reshape(1, -1))
    top_three = [sorted(zip(model.classes_, prob), key=lambda x: x[1], reverse=True)[:3] for prob in probabilities]
    
    return top_three


def recommendation(N,P,K,temperature,humidity,ph,rainfall,model):
    features=np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    result=[]

    top_three = prediction_model(features,model)
    for i in top_three[0]:
        result.append(i[0])
    
    return result
# ============================================================================================================



# python main
if __name__ == "__main__":
    app.run(debug=True)

