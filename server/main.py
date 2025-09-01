from fastapi import FastAPI, Depends, HTTPException, status, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import pymongo
import os
from dotenv import load_dotenv
from typing import List, Dict, Any


load_dotenv()

app = FastAPI()

# --- CORS ---
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://116.202.210.102:3005")
# configure allowed origins (no wildcard when credentials=True)
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
allowed_origins = [
    FRONTEND_URL,
    "http://116.202.210.102:3005",
    
    
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:3000",
    "http://127.0.0.1:3000",  # add thi
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in allowed_origins if origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ensure_cors_headers(request, call_next):
    # Return quick response for OPTIONS preflight
    if request.method == "OPTIONS":
        origin = request.headers.get("origin") or ""
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Headers": "Authorization,Content-Type",
            "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
        }
        return Response(status_code=200, headers=headers)

    response = await call_next(request)
    origin = request.headers.get("origin")
    if origin:
        # explicitly echo origin to allow credentials from browser
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.on_event("startup")
def startup_db():
    # Ensure unique index on email to prevent duplicate users
    try:
        users_collection.create_index("email", unique=True)
    except Exception as e:
        # Index might already exist or connection issue; log in real app
        print("startup index creation error:", e)

# --- MongoDB Connection ---
# replaced manual connection with shared db module
from db import db, users_collection, searches_collection


# --- JWT Settings ---
SECRET_KEY = os.getenv("SECRET_KEY", "secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# --- Password Hashing ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Models ---
class User(BaseModel):
    username: str
    email: str
    password: str

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    
class InfluencerSearch(BaseModel):
    keyword: str
    results: List[Dict[str, Any]]
    created_at: datetime = datetime.utcnow()

# --- Utility Functions ---
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user(email: str):
    return users_collection.find_one({"email": email})

def authenticate_user(email: str, password: str):
    user = get_user(email)
    if not user or not verify_password(password, user["password"]):
        return False
    return user


class LoginModel(BaseModel):
    email: str
    password: str

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials"
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(email)
    if user is None:
        raise credentials_exception
    return user


from influencers import router as influencers_router
from auth import router as auth_router, get_current_user as auth_get_current_user

app.include_router(influencers_router, prefix="/influencers")
# auth_router already defines its own prefix (`/auth`) in `server/auth.py`,
# so include it without adding another `/auth` prefix to avoid double routes.
app.include_router(auth_router)

@app.get("/me")
def read_users_me(current_user: dict = Depends(auth_get_current_user)):
    return {"username": current_user["username"], "email": current_user["email"]}

