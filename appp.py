

from flask import Flask, render_template, request
import joblib  
import pandas as pd


app = Flask(__name__)


model = joblib.load('mlop.pkl')  


original_columns = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
    'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
    'parking', 'prefarea', 'furnishingstatus'
]

def preprocess_input(data):
    """
    Preprocess the input data to match the format required by the model.
    """
   
    df = pd.DataFrame([data])

    
    df_final = df.reindex(columns=original_columns, fill_value=0)

    return df_final


# Flask route to handle prediction
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handle the prediction request and render the result.
    """
    price = None
    if request.method == "POST":
        data = {
            'area': float(request.form['area']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'stories': int(request.form['stories']),
            'mainroad': int(request.form['mainroad']),
            'guestroom': int(request.form['guestroom']),
            'basement': int(request.form['basement']),
            'hotwaterheating': int(request.form['hotwaterheating']),
            'airconditioning': int(request.form['airconditioning']),
            'parking': int(request.form['parking']),
            'prefarea': int(request.form['prefarea']),
            'furnishingstatus': int(request.form['furnishingstatus'])
        }

        preprocessed_data = preprocess_input(data)
        prediction = model.predict(preprocessed_data)[0]
        price = f"${prediction:,.2f}"

    return render_template('index.html', predicted_price=price)

if __name__ == "__main__":
    app.run(debug=True)
