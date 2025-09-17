"""
Multi-User FastAPI Service for Intent Recognition with Vectorization and Semantic Search
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === MODELS ===
class UserCreate(BaseModel):
    username: str
    
    @validator('username')
    def validate_username(cls, v):
        if not v.strip():
            raise ValueError('Username cannot be empty')
        if len(v.strip()) < 3:
            raise ValueError('Username must be at least 3 characters')
        # Remove special characters that could cause file system issues
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v.strip().lower()

class UserLogin(BaseModel):
    username: str

class ProjectCreate(BaseModel):
    username: str
    project_name: str
    
    @validator('project_name')
    def validate_project_name(cls, v):
        if not v.strip():
            raise ValueError('Project name cannot be empty')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Project name can only contain letters, numbers, underscores, and hyphens')
        return v.strip().lower()

class IntentData(BaseModel):
    language: str
    project_name: str
    intent_trajectory: Dict[str, List[str]]

class IntentUpload(BaseModel):
    username: str
    intent_data: IntentData

class EmbeddingCreate(BaseModel):
    username: str
    project_name: str

class EmbeddingUpdate(BaseModel):
    username: str
    project_name: str
    intent_name: str
    examples: List[str]

class EmbeddingDelete(BaseModel):
    username: str
    project_name: str
    intent_name: str

class IntentPredict(BaseModel):
    username: str
    project_name: str
    text: str

# === EMBEDDING MODEL MANAGER ===
class EmbeddingModelManager:
    """Singleton pattern for managing the embedding model in memory"""
    
    _instance = None
    _model = None
    _tokenizer = None
    _model_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Load embedding model (lazy loading)"""
        if not self._model_loaded:
            try:
                # Using sentence-transformers for better semantic understanding
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model...")
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                self._model_loaded = True
                logger.info("Embedding model loaded successfully")
            except ImportError:
                # Fallback to a simple TF-IDF approach if sentence-transformers is not available
                logger.warning("sentence-transformers not available, using fallback embedding")
                from sklearn.feature_extraction.text import TfidfVectorizer
                self._model = TfidfVectorizer(max_features=384, stop_words='english')
                self._model_loaded = True
        
        return self._model
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        model = self.load_model()
        
        if hasattr(model, 'encode'):
            # sentence-transformers model
            return model.encode(texts)
        else:
            # TF-IDF fallback
            if hasattr(model, 'transform'):
                return model.transform(texts).toarray()
            else:
                return model.fit_transform(texts).toarray()
    
    def free_model_memory(self):
        """Free model from memory when not needed"""
        if self._model_loaded:
            self._model = None
            self._model_loaded = False
            logger.info("Model memory freed")

# === VECTOR DATABASE ===
class LocalVectorDB:
    """Local vector database for storing and searching embeddings"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.embeddings = {}
        self.intent_labels = {}
        self.load_db()
    
    def get_db_file(self, username: str, project_name: str) -> Path:
        """Get database file path for user project"""
        return self.db_path / f"{username}_{project_name}_vectors.pkl"
    
    def load_db(self):
        """Load existing vector database"""
        try:
            for db_file in self.db_path.glob("*_vectors.pkl"):
                key = db_file.stem.replace("_vectors", "")
                with open(db_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings[key] = data.get('embeddings', {})
                    self.intent_labels[key] = data.get('intent_labels', {})
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
    
    def save_db(self, username: str, project_name: str):
        """Save vector database to disk"""
        key = f"{username}_{project_name}"
        db_file = self.get_db_file(username, project_name)
        
        data = {
            'embeddings': self.embeddings.get(key, {}),
            'intent_labels': self.intent_labels.get(key, {}),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'username': username,
                'project_name': project_name
            }
        }
        
        with open(db_file, 'wb') as f:
            pickle.dump(data, f)
    
    def create_vectors(self, username: str, project_name: str, intent_data: Dict[str, List[str]]):
        """Create vectors for intents"""
        key = f"{username}_{project_name}"
        
        # Initialize storage
        self.embeddings[key] = {}
        self.intent_labels[key] = {}
        
        # Get embedding model
        embedding_manager = EmbeddingModelManager()
        
        # Process each intent
        for intent_name, examples in intent_data.items():
            if examples:
                # Generate embeddings for all examples
                embeddings = embedding_manager.encode(examples)
                
                # Store embeddings and labels
                self.embeddings[key][intent_name] = embeddings
                self.intent_labels[key][intent_name] = examples
        
        # Save to disk
        self.save_db(username, project_name)
        
        # Free model memory after use
        embedding_manager.free_model_memory()
    
    def update_intent_vectors(self, username: str, project_name: str, intent_name: str, examples: List[str]):
        """Update vectors for a specific intent"""
        key = f"{username}_{project_name}"
        
        if key not in self.embeddings:
            raise ValueError("Project vectors not found")
        
        # Get embedding model
        embedding_manager = EmbeddingModelManager()
        
        # Generate new embeddings
        embeddings = embedding_manager.encode(examples)
        
        # Update storage
        self.embeddings[key][intent_name] = embeddings
        self.intent_labels[key][intent_name] = examples
        
        # Save to disk
        self.save_db(username, project_name)
        
        # Free model memory
        embedding_manager.free_model_memory()
    
    def delete_intent_vectors(self, username: str, project_name: str, intent_name: str):
        """Delete vectors for a specific intent"""
        key = f"{username}_{project_name}"
        
        if key in self.embeddings and intent_name in self.embeddings[key]:
            del self.embeddings[key][intent_name]
            del self.intent_labels[key][intent_name]
            self.save_db(username, project_name)
            return True
        
        return False
    
    def search_similar(self, username: str, project_name: str, query_text: str, top_k: int = 1) -> Dict[str, float]:
        """Search for similar intents"""
        key = f"{username}_{project_name}"
        
        if key not in self.embeddings or not self.embeddings[key]:
            raise ValueError("No vectors found for this project")
        
        # Get embedding model and encode query
        embedding_manager = EmbeddingModelManager()
        query_embedding = embedding_manager.encode([query_text])
        
        # Calculate similarities with all intents
        similarities = {}
        
        for intent_name, intent_embeddings in self.embeddings[key].items():
            # Calculate cosine similarity with all examples of this intent
            sims = cosine_similarity(query_embedding, intent_embeddings)[0]
            # Take the maximum similarity score for this intent
            similarities[intent_name] = float(np.max(sims))
        
        # Free model memory
        embedding_manager.free_model_memory()
        
        # Sort by similarity and return top results
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_similarities[:top_k])

# === FILE SYSTEM MANAGER ===
class FileSystemManager:
    """Manage user and project file structure"""
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def create_user_folder(self, username: str) -> bool:
        """Create user folder"""
        user_path = self.base_path / username
        if user_path.exists():
            return False  # User already exists
        
        user_path.mkdir()
        logger.info(f"Created user folder: {user_path}")
        return True
    
    def user_exists(self, username: str) -> bool:
        """Check if user exists"""
        return (self.base_path / username).exists()
    
    def create_project_folder(self, username: str, project_name: str) -> bool:
        """Create project folder under user"""
        if not self.user_exists(username):
            return False
        
        project_path = self.base_path / username / project_name
        if project_path.exists():
            return False  # Project already exists
        
        project_path.mkdir()
        logger.info(f"Created project folder: {project_path}")
        return True
    
    def project_exists(self, username: str, project_name: str) -> bool:
        """Check if project exists"""
        return (self.base_path / username / project_name).exists()
    
    def save_intent_data(self, username: str, project_name: str, intent_data: Dict[str, Any]):
        """Save intent data to JSON file"""
        project_path = self.base_path / username / project_name
        intent_file = project_path / "intents.json"
        
        with open(intent_file, 'w') as f:
            json.dump(intent_data, f, indent=2)
        
        logger.info(f"Saved intent data: {intent_file}")
    
    def load_intent_data(self, username: str, project_name: str) -> Optional[Dict[str, Any]]:
        """Load intent data from JSON file"""
        project_path = self.base_path / username / project_name
        intent_file = project_path / "intents.json"
        
        if not intent_file.exists():
            return None
        
        with open(intent_file, 'r') as f:
            return json.load(f)
    
    def get_user_count(self) -> int:
        """Get total number of users"""
        return len([d for d in self.base_path.iterdir() if d.is_dir()])
    
    def get_project_count(self) -> int:
        """Get total number of projects across all users"""
        total_projects = 0
        for user_dir in self.base_path.iterdir():
            if user_dir.is_dir():
                total_projects += len([p for p in user_dir.iterdir() if p.is_dir()])
        return total_projects

# === GLOBAL INSTANCES ===
fs_manager = FileSystemManager()
vector_db = LocalVectorDB("./vector_db")

# === LIFESPAN MANAGEMENT ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Intent Recognition Service...")
    yield
    # Shutdown
    logger.info("Shutting down Intent Recognition Service...")
    # Free model memory on shutdown
    embedding_manager = EmbeddingModelManager()
    embedding_manager.free_model_memory()

# === FASTAPI APP ===
app = FastAPI(
    title="Intent Analysis & Semantic Search Platform",
    description="A multi-user platform for managing intent analysis projects with vector-based semantic search and high-confidence predictions.",
    version="1.0.0",
    lifespan=lifespan
)

# === DEPENDENCY FUNCTIONS ===
def validate_user_exists(username: str):
    """Validate that user exists"""
    if not fs_manager.user_exists(username):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "User not found", "status": 404}
        )
    return username

def validate_project_exists(username: str, project_name: str):
    """Validate that project exists for user"""
    validate_user_exists(username)
    if not fs_manager.project_exists(username, project_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Project not found", "status": 404}
        )
    return username, project_name

# === API ENDPOINTS ===

# User Management
@app.post("/user/create")
async def create_user(user_data: UserCreate):
    """Create a new user account"""
    try:
        if fs_manager.create_user_folder(user_data.username):
            return {
                "message": "User created successfully",
                "username": user_data.username,
                "status": 201
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": "User already exists", "status": 409}
            )
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "status": 500}
        )

@app.post("/user/login")
async def login_user(user_data: UserLogin):
    """Validate user login"""
    try:
        validate_user_exists(user_data.username)
        return {
            "message": "Login successful",
            "username": user_data.username,
            "status": 200
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "status": 500}
        )

# Project Management
@app.post("/project/create")
async def create_project(project_data: ProjectCreate):
    """Create a new project for user"""
    try:
        validate_user_exists(project_data.username)
        
        if fs_manager.create_project_folder(project_data.username, project_data.project_name):
            return {
                "message": "Project created successfully",
                "username": project_data.username,
                "project_name": project_data.project_name,
                "status": 201
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": "Project already exists", "status": 409}
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "status": 500}
        )

@app.post("/project/upload_intents")
async def upload_intents(upload_data: IntentUpload):
    """Upload intent data to project"""
    try:
        validate_project_exists(upload_data.username, upload_data.intent_data.project_name)
        
        # Save intent data to file
        fs_manager.save_intent_data(
            upload_data.username,
            upload_data.intent_data.project_name,
            upload_data.intent_data.dict()
        )
        
        return {
            "message": "Intent data uploaded successfully",
            "username": upload_data.username,
            "project_name": upload_data.intent_data.project_name,
            "intents": list(upload_data.intent_data.intent_trajectory.keys()),
            "status": 200
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading intents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "status": 500}
        )

# Embedding Management
@app.post("/embedding/create")
async def create_embeddings(embedding_data: EmbeddingCreate):
    """Create embeddings for project intents"""
    try:
        validate_project_exists(embedding_data.username, embedding_data.project_name)
        
        # Load intent data
        intent_data = fs_manager.load_intent_data(embedding_data.username, embedding_data.project_name)
        if not intent_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Intent data not found. Upload intents first.", "status": 404}
            )
        
        # Create vectors
        vector_db.create_vectors(
            embedding_data.username,
            embedding_data.project_name,
            intent_data['intent_trajectory']
        )
        
        return {
            "message": "Embeddings created successfully",
            "username": embedding_data.username,
            "project_name": embedding_data.project_name,
            "intents": list(intent_data['intent_trajectory'].keys()),
            "status": 200
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "status": 500}
        )

@app.post("/embedding/update")
async def update_embeddings(update_data: EmbeddingUpdate):
    """Update embeddings for specific intent"""
    try:
        validate_project_exists(update_data.username, update_data.project_name)
        
        vector_db.update_intent_vectors(
            update_data.username,
            update_data.project_name,
            update_data.intent_name,
            update_data.examples
        )
        
        return {
            "message": "Embeddings updated successfully",
            "username": update_data.username,
            "project_name": update_data.project_name,
            "intent_name": update_data.intent_name,
            "status": 200
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e), "status": 404}
        )
    except Exception as e:
        logger.error(f"Error updating embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "status": 500}
        )

@app.delete("/embedding/delete")
async def delete_embeddings(delete_data: EmbeddingDelete):
    """Delete embeddings for specific intent"""
    try:
        validate_project_exists(delete_data.username, delete_data.project_name)
        
        if vector_db.delete_intent_vectors(
            delete_data.username,
            delete_data.project_name,
            delete_data.intent_name
        ):
            return {
                "message": "Embeddings deleted successfully",
                "username": delete_data.username,
                "project_name": delete_data.project_name,
                "intent_name": delete_data.intent_name,
                "status": 200
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "Intent not found", "status": 404}
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "status": 500}
        )

# Search
@app.post("/intent/predict")
async def predict_intent(predict_data: IntentPredict):
    """Predict intent for given text"""
    try:
        validate_project_exists(predict_data.username, predict_data.project_name)
        
        # Search for similar intents
        similarities = vector_db.search_similar(
            predict_data.username,
            predict_data.project_name,
            predict_data.text
        )
        
        if not similarities:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "status": 200
            }
        
        # Get top prediction
        top_intent, confidence = next(iter(similarities.items()))
        
        return {
            "intent": top_intent,
            "confidence": round(confidence, 2),
            "status": 200
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": str(e), "status": 404}
        )
    except Exception as e:
        logger.error(f"Error predicting intent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "status": 500}
        )

# System Info
@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        user_count = fs_manager.get_user_count()
        project_count = fs_manager.get_project_count()
        
        return {
            "total_users": user_count,
            "total_projects": project_count,
            "status": 200
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Internal server error", "status": 500}
        )

# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
