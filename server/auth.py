# auth.py  — FastAPI version matching the original Flask routes/behavior

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from bson import ObjectId
import pymongo
import re
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/auth", tags=["Auth"])

# ----------------- MongoDB -----------------
client = pymongo.MongoClient(os.getenv("MONGO_URI"))
db = client["influencer_db"]
users = db["users"]

# Optional: ensure uniqueness at DB level
# (runs once; harmless if indexes already exist)
users.create_index("email", unique=True)
users.create_index("username", unique=True)

# ----------------- JWT -----------------
ACCESS_SECRET = os.getenv("ACCESS_SECRET", os.getenv("SECRET_KEY", "access_secret"))
REFRESH_SECRET = os.getenv("REFRESH_SECRET", os.getenv("SECRET_KEY", "refresh_secret"))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", 7))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")  # used for access token

# ----------------- Password hashing -----------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ----------------- Helpers -----------------
def is_strong_password(password: str) -> bool:
    # At least 8 chars, one uppercase, one special char, one number
    if (
        len(password) < 8
        or not re.search(r"[A-Z]", password)
        or not re.search(r"[^A-Za-z0-9]", password)
        or not re.search(r"\d", password)
    ):
        return False
    return True

def create_access_token(sub: str) -> str:
    payload = {
        "sub": sub,
        "typ": "access",
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, ACCESS_SECRET, algorithm=ALGORITHM)

def create_refresh_token(sub: str) -> str:
    payload = {
        "sub": sub,
        "typ": "refresh",
        "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, REFRESH_SECRET, algorithm=ALGORITHM)

def get_user_by_email(email: str):
    return users.find_one({"email": email})

def get_user_by_id(user_id: str):
    try:
        return users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        return None

async def get_current_user(token: str = Depends(oauth2_scheme)):
    # Access-token protected dependency (like @jwt_required())
    try:
        payload = jwt.decode(token, ACCESS_SECRET, algorithms=[ALGORITHM])
        if payload.get("typ") != "access":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")
        uid = payload.get("sub")
        if not uid:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = get_user_by_id(uid)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

# ----------------- Schemas -----------------
class RegisterIn(BaseModel):
    username: str = Field(..., min_length=3)
    email: EmailStr
    password: str
    confirm_password: str
    first_name: str
    last_name: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: str
    username: str
    email: EmailStr
    first_name: str
    last_name: str

class UpdateIn(BaseModel):
    username: str | None = None
    email: EmailStr | None = None
    first_name: str | None = None
    last_name: str | None = None
    current_password: str | None = None
    new_password: str | None = None



@router.post("/register", status_code=201)
def register(data: RegisterIn):
    if get_user_by_email(data.email):
        # 409 in Flask for "exists"; we'll mirror that
        raise HTTPException(status_code=409, detail="Email already exists")
    if users.find_one({"username": data.username}):
        raise HTTPException(status_code=409, detail="Username already exists")
    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if not is_strong_password(data.password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long, contain one uppercase letter, one special character, and one number.",
        )

    doc = {
        "username": data.username,
        "email": data.email,
        "first_name": data.first_name,
        "last_name": data.last_name,
        "password": hash_password(data.password),
    }
    result = users.insert_one(doc)
    # After creating user, issue tokens (same shape as /login)
    uid = str(result.inserted_id)
    access_token = create_access_token(uid)
    refresh_token = create_refresh_token(uid)

    return {
        "user": {
            "id": uid,
            "username": data.username,
            "email": data.email,
            "first_name": data.first_name,
            "last_name": data.last_name,
        },
        "access_token": access_token,
        "refresh_token": refresh_token,
    }

@router.post("/login")
def login(body: LoginIn):
    print("LOGIN BODY:", body)
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    uid = str(user["_id"])
    access_token = create_access_token(uid)
    refresh_token = create_refresh_token(uid)

    # Return same shape as Flask: user + tokens
    return {
        "user": {
            "id": uid,
            "username": user["username"],
            "email": user["email"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
        },
        "access_token": access_token,
        "refresh_token": refresh_token,
    }

@router.post("/refresh")
def refresh(request: Request):
    
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing refresh token")
    token = auth.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, REFRESH_SECRET, algorithms=[ALGORITHM])
        if payload.get("typ") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        uid = payload.get("sub")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    new_access = create_access_token(uid)
    return {"access_token": new_access}

@router.get("/user", response_model=UserOut)
async def get_user(current_user = Depends(get_current_user)):
    return UserOut(
        id=str(current_user["_id"]),
        username=current_user["username"],
        email=current_user["email"],
        first_name=current_user["first_name"],
        last_name=current_user["last_name"],
    )

@router.patch("/user")
async def update_user(data: UpdateIn, current_user = Depends(get_current_user)):
    uid = current_user["_id"]
    updates: dict = {}

    # username/email uniqueness checks (excluding current user)
    if data.username and data.username != current_user["username"]:
        if users.find_one({"username": data.username, "_id": {"$ne": uid}}):
            raise HTTPException(status_code=409, detail="Username already taken")
        updates["username"] = data.username

    if data.email and data.email != current_user["email"]:
        if users.find_one({"email": data.email, "_id": {"$ne": uid}}):
            raise HTTPException(status_code=409, detail="Email already taken")
        updates["email"] = data.email

    if data.first_name is not None:
        updates["first_name"] = data.first_name
    if data.last_name is not None:
        updates["last_name"] = data.last_name

    # password change
    if data.new_password:
        if not data.current_password or not verify_password(data.current_password, current_user["password"]):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        if not is_strong_password(data.new_password):
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters long, contain one uppercase letter, one special character, and one number.",
            )
        updates["password"] = hash_password(data.new_password)

    if updates:
        users.update_one({"_id": uid}, {"$set": updates})

    return {"message": "Profile updated successfully"}

@router.delete("/user")
async def delete_user(current_user = Depends(get_current_user)):
    users.delete_one({"_id": current_user["_id"]})
    return {"message": "Account deleted"}
