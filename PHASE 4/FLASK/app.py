import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

#create flask app
app = Flask(__name__)
#Load the Pickle Model
model=pickle.load(open("model.pkl","rb"))
ms=pickle.load(open("ms.pkl","rb"))
le=pickle.load(open("le.pkl","rb"))
@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/guest", methods=["POST"])
def Guest():
    Airline_name = request.form['Airline_name']  # Fix variable name
    Seat_Type = request.form['Seat_Type']
    Type_Of_Traveller = request.form['Type_Of_Traveller']
    Origin = request.form['Origin']
    Destination = request.form['Destination']
    Month_Flown = request.form['Month_Flown']
    Year_Flown = request.form['Year_Flown']
    Verified = request.form['Verified']
    Seat_Comfort = request.form['Seat_Comfort']
    Food_Beverages = request.form['Food_Beverages']  # Fix variable name
    Ground_Service = request.form['Ground_Service']
    O_R = request.form['O_R']

    data = [[Airline_name, Seat_Type, Type_Of_Traveller, Origin,
             Destination, Month_Flown, Year_Flown, Verified, Seat_Comfort, Food_Beverages, Ground_Service, O_R]]

    encoded_data = [
        le.transform([Airline_name])[0],
        le.transform([Seat_Type])[0],
        le.transform([Type_Of_Traveller])[0],
        le.transform([Origin])[0],
        le.transform([Destination])[0],
        le.transform([Month_Flown])[0],
        le.transform([Year_Flown])[0],
        le.transform([Verified])[0],
        le.transform([Seat_Comfort])[0],  
        le.transform([Food_Beverages])[0],  
        le.transform([Ground_Service])[0],
        le.transform([O_R])[0]
    ]

    print(encoded_data)

    prediction = model.predict(ms.transform([encoded_data]))

    if prediction == 1:
        a = "Recommended"
        return render_template('index.html', y=a)
    else:
        b = "Not Recommended"
        return render_template('index.html', y=b)
if __name__ =="__main__":
    app.run(debug=True)