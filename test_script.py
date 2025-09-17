import requests

# 1. Create user
response = requests.post("http://localhost:8000/user/create", 
    json={"username": "alice"})

# 2. Create project
response = requests.post("http://localhost:8000/project/create", 
    json={"username": "alice", "project_name": "your_bot"})

# 3. Upload intents
intent_data = {
    "username": "alice",
    "intent_data": {
        "language": "en-IN",
        "project_name": "your_bot",
        "intent_trajectory": {
            "Yes": ["yes", "yep", "sure", "absolutely"],
            "No": ["no", "not", "never", "nah"],
            "Neutral": ["maybe", "possibly", "not sure"]
        }
    }
}
response = requests.post("http://localhost:8000/project/upload_intents", json=intent_data)

# 4. Create embeddings
response = requests.post("http://localhost:8000/embedding/create", 
    json={"username": "alice", "project_name": "your_bot"})

# 5. Predict intent
response = requests.post("http://localhost:8000/intent/predict", 
    json={"username": "alice", "project_name": "your_bot", "text": "sure absolutely"})
# Returns: {"intent": "Yes", "confidence": 0.92}