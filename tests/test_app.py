# tests/test_app.py

from fastapi.testclient import TestClient
# Make sure your FastAPI app instance is accessible to be imported
from src.app import app

# Create a client to interact with the app
client = TestClient(app)

def test_read_root():
    """
    Tests that the root endpoint returns a 200 OK status and the correct JSON message.
    """
    # Send a GET request to the "/" endpoint
    response = client.get("/")
    
    # Assert that the HTTP status code is 200 (OK)
    assert response.status_code == 200
    
    # Assert that the response body is the expected JSON
    assert response.json() == {"message": "Welcome to the Iris Classifier API. Go to /docs for documentation."}