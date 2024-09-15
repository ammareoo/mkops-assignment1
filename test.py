import pytest
import joblib
import os
from flask import Flask
from app import app as flask_app

# Load the model for testing
model = joblib.load('mlop.pkl')

# Test if the model file exists
def test_model_exists():
    assert os.path.exists('mlop.pkl'), "Model file not found!"

# Test the model prediction (use some sample input similar to training data)
def test_model_prediction():
    test_input = [[5000, 4, 3, 2, 1, 0, 1, 0, 1, 2, 1, 0]]
    prediction = model.predict(test_input)
    assert prediction is not None, "Prediction should not be None"
    assert prediction[0] > 0, "Prediction should be a positive value"

# API Test client setup
@pytest.fixture
def client():
    with flask_app.test_client() as client:
        yield client

# Test if the home route is accessible
def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200, "Home route should be accessible"

# Test if the predict route handles POST requests correctly
def test_predict_route(client):
    test_data = {
        'area': 5000,
        'bedrooms': 4,
        'bathrooms': 3,
        'stories': 2,
        'mainroad': 1,
        'guestroom': 0,
        'basement': 1,
        'hotwaterheating': 0,
        'airconditioning': 1,
        'parking': 2,
        'prefarea': 1,
        'furnishingstatus': 0
    }
    response = client.post('/predict', json=test_data)
    assert response.status_code == 200, "Prediction route should return 200"
    json_data = response.get_json()
    assert 'predicted_price' in json_data, "Response should contain predicted price"
    assert json_data['predicted_price'] > 0, "Predicted price should be a positive value"
