import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,render_template

# ======================================================================
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")
# =========================================================================

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    

    prediction = recommendation(N, P, K, temp, humidity, ph, rainfall)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)


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



def prediction_model(df, features):
    

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
    
#     ypred = rfc.predict(x_test)
#     accuracy = accuracy_score(y_test, ypred)

    prediction = rfc.predict(features.reshape(1, -1))
    return prediction[0]


def recommendation(N,P,K,temperature,humidity,ph,rainfall):
    features=np.array([[N,P,K,temperature,humidity,ph,rainfall]])
    result=[]
    df = crop.copy()  # Make a copy of the DataFrame to avoid modifying the original data
    df['crop_num'] = df['label'].map(crop_dict)
    df.drop('label', axis=1, inplace=True)
    for i in range(1):
        val=prediction_model(df,features)
        result.append(val)
        print(val)
        df=df[df['crop_num'] != val]
    
    return result
# ============================================================================================================



# python main
if __name__ == "__main__":
    app.run(debug=True)

