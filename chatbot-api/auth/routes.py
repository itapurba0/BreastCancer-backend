from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from database import users_collection
from auth.deps import hash_password, verify_password, create_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


class AuthRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    email: str


@router.post("/signup", response_model=AuthResponse)
def signup(body: AuthRequest):
    email = body.email.strip().lower()
    if not email or not body.password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    if users_collection.find_one({"email": email}):
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed = hash_password(body.password)
    users_collection.insert_one({"email": email, "hashed_password": hashed, "created_at": datetime.now(timezone.utc)})

    token = create_token(email)
    return AuthResponse(token=token, email=email)


@router.post("/login", response_model=AuthResponse)
def login(body: AuthRequest):
    email = body.email.strip().lower()
    user = users_collection.find_one({"email": email})
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(email)
    return AuthResponse(token=token, email=email)


@router.get("/me")
def me(email: str = Depends(get_current_user)):
    return {"email": email}
