# 🤖 Multi-User Intent Analysis & Semantic Search Platform

A production-ready **FastAPI service** for managing intent recognition projects with **vectorization** and **semantic search**. Built for multi-user environments with complete data isolation, efficient memory management, and local vector database storage.

## 🌟 Features

### 🔐 **Multi-User Support**
- **Complete data isolation** - Users can't access each other's data
- **Persistent storage** - Data survives server restarts and user sessions
- **User-specific folders** - Organized file structure for each user

### 🗂️ **Project Management**
- **Multiple projects per user** with isolated storage
- **Intent definition storage** via JSON files
- **Local vector database** for each project (no external dependencies)

### 🧠 **Smart Vectorization**
- **Shared embedding model** across all users for efficiency
- **Memory-optimized** - Model loads only when needed, frees memory after use
- **Semantic search** using cosine similarity
- **High-quality embeddings** with sentence-transformers (TF-IDF fallback)

### 🔍 **Intent Recognition**
- **Real-time prediction** with confidence scores
- **Vector similarity search** for accurate intent matching
- **CRUD operations** for intent management (Create, Read, Update, Delete)

### 📸 **Intent Recognition Screenshot**

<p align="center">
  <img src="https://github.com/SushilkumarBarai/Semantic_Intent_Analysis/blob/main/Screenshot.png" alt="Semantic Screenshot" width="600"/>
</p>

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.9+
pip package manager
```

### Installation

1. **Clone or create the project**
```bash
mkdir intent-recognition-service
cd intent-recognition-service
```

2. **Install dependencies**
```bash
pip install fastapi uvicorn pydantic numpy scikit-learn sentence-transformers
```

3. **Save the main service code as `main.py`**
   
4. **Run the service**
```bash
python main.py
```

The service will start on `http://localhost:8000`

### 📖 API Documentation
Once running, visit:
- **Interactive docs**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 API Endpoints

### 👤 User Management
```http
POST /user/create          # Create new user account
POST /user/login           # Validate user login
```

### 📁 Project Management
```http
POST /project/create       # Create new project
POST /project/upload_intents # Upload intent definitions
```

### 🧮 Embedding Management
```http
POST /embedding/create     # Generate vectors for all intents
POST /embedding/update     # Update specific intent vectors
DELETE /embedding/delete   # Delete intent vectors
```

### 🔍 Intent Prediction
```http
POST /intent/predict       # Predict intent from text
```

### 📊 System Information
```http
GET /stats                 # Get user and project counts
GET /health               # Health check
```

## 📝 Usage Example

### Step 1: Create a User
```bash
curl -X POST "http://localhost:8000/user/create" \
     -H "Content-Type: application/json" \
     -d '{"username": "alice"}'
```

**Response:**
```json
{
  "message": "User created successfully",
  "username": "alice",
  "status": 201
}
```

### Step 2: Create a Project
```bash
curl -X POST "http://localhost:8000/project/create" \
     -H "Content-Type: application/json" \
     -d '{"username": "alice", "project_name": "chatbot_v1"}'
```

### Step 3: Upload Intent Data
```bash
curl -X POST "http://localhost:8000/project/upload_intents" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "alice",
       "intent_data": {
         "language": "en-US",
         "project_name": "chatbot_v1",
         "intent_trajectory": {
           "Yes": ["yes", "yep", "sure", "absolutely", "of course"],
           "No": ["no", "not", "never", "nah", "nope"],
           "Greeting": ["hello", "hi", "hey", "good morning"],
           "Goodbye": ["bye", "goodbye", "see you later", "farewell"]
         }
       }
     }'
```

### Step 4: Generate Embeddings
```bash
curl -X POST "http://localhost:8000/embedding/create" \
     -H "Content-Type: application/json" \
     -d '{"username": "alice", "project_name": "chatbot_v1"}'
```

### Step 5: Predict Intent
```bash
curl -X POST "http://localhost:8000/intent/predict" \
     -H "Content-Type: application/json" \
     -d '{"username": "alice", "project_name": "chatbot_v1", "text": "sure absolutely"}'
```

**Response:**
```json
{
  "intent": "Yes",
  "confidence": 0.92,
  "status": 200
}
```

## 🗂️ File Structure

The service creates the following structure:
```
.
├── main.py                 # Main FastAPI application
├── data/                   # User data storage
│   ├── alice/              # User folder
│   │   ├── chatbot_v1/     # Project folder
│   │   │   └── intents.json # Intent definitions
│   │   └── shopping_bot/   # Another project
│   └── bob/                # Another user
└── vector_db/              # Vector database storage
    ├── alice_chatbot_v1_vectors.pkl
    └── bob_assistant_vectors.pkl
```

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom data directory
export DATA_DIR="./custom_data"

# Optional: Set custom vector DB directory  
export VECTOR_DB_DIR="./custom_vectors"

# Optional: Change server host/port
export HOST="0.0.0.0"
export PORT="8000"
```

### Memory Management
The service automatically manages memory by:
- Loading the embedding model only when needed
- Freeing model memory after generating embeddings
- Using efficient vector storage with pickle serialization

## 📊 Error Handling

The service provides comprehensive error responses:

```json
// User not found
{
  "error": "User not found",
  "status": 404
}

// Project not found
{
  "error": "Project not found", 
  "status": 404
}

// User already exists
{
  "error": "User already exists",
  "status": 409
}
```

## 🔒 Data Security & Isolation

- **Complete user isolation** - Users cannot access other users' data
- **Folder-based separation** - Each user has their own directory
- **Project-level isolation** - Projects are stored in separate subfolders
- **Vector database separation** - Each project has its own vector file

## ⚡ Performance Features

### Memory Efficiency
- **Singleton pattern** for embedding model (shared across all users)
- **Lazy loading** - Model loads only when generating embeddings
- **Automatic cleanup** - Frees model memory after use

### Storage Optimization
- **Local vector database** using pickle for fast serialization
- **Efficient similarity search** with cosine similarity
- **Minimal memory footprint** for vector storage

### Scalability
- **File-based storage** scales with disk space
- **Concurrent user support** with proper isolation
- **Stateless design** for horizontal scaling

## 🛠️ Advanced Usage

### Update Specific Intent
```bash
curl -X POST "http://localhost:8000/embedding/update" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "alice",
       "project_name": "chatbot_v1", 
       "intent_name": "Yes",
       "examples": ["yes", "yep", "sure", "absolutely", "definitely", "for sure"]
     }'
```

### Delete Intent
```bash
curl -X DELETE "http://localhost:8000/embedding/delete" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "alice",
       "project_name": "chatbot_v1",
       "intent_name": "Greeting"
     }'
```

### Get System Statistics
```bash
curl -X GET "http://localhost:8000/stats"
```

**Response:**
```json
{
  "total_users": 5,
  "total_projects": 12,
  "status": 200
}
```

## 🐳 Docker Support

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  intent-service:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./vector_db:/app/vector_db
    environment:
      - HOST=0.0.0.0
      - PORT=8000
```

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Load Testing Example
```python
import asyncio
import aiohttp
import time

async def test_prediction():
    async with aiohttp.ClientSession() as session:
        payload = {
            "username": "alice",
            "project_name": "chatbot_v1", 
            "text": "absolutely yes"
        }
        async with session.post('http://localhost:8000/intent/predict', json=payload) as resp:
            return await resp.json()

# Run multiple concurrent requests
start_time = time.time()
results = await asyncio.gather(*[test_prediction() for _ in range(100)])
end_time = time.time()
print(f"100 predictions in {end_time - start_time:.2f} seconds")
```

## 🔍 Troubleshooting

### Common Issues

1. **Module not found errors**
   ```bash
   pip install sentence-transformers
   # or use fallback mode (TF-IDF will be used automatically)
   ```

2. **Permission errors on folder creation**
   ```bash
   chmod 755 ./data
   chmod 755 ./vector_db
   ```

3. **Memory issues with large models**
   - The service automatically manages memory
   - Model is freed after each embedding generation
   - Consider increasing system RAM for very large datasets

4. **Port already in use**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

## 📈 Production Deployment

### Recommended Enhancements
- **Authentication**: Add JWT token authentication
- **Database**: Use PostgreSQL for metadata storage
- **Caching**: Implement Redis for frequently accessed vectors
- **Monitoring**: Add Prometheus metrics and logging
- **Rate Limiting**: Implement API rate limiting
- **Load Balancing**: Use Nginx for multiple service instances

### Environment Setup
```bash
# Production requirements
pip install gunicorn[gevent]

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```


**Built with ❤️ using FastAPI, sentence-transformers, and scikit-learn**
