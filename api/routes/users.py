"""
api/routes/users.py — User registration, login, profile, password reset
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from db.database import get_db
from db.models import User
from api.auth import hash_password, verify_password, create_access_token, get_current_user, Token

router = APIRouter(prefix="/api/users", tags=["users"])


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool

    class Config:
        from_attributes = True


class ForgotPasswordRequest(BaseModel):
    email: str


class VerifyOTPRequest(BaseModel):
    email: str
    otp: str


class ResetPasswordRequest(BaseModel):
    email: str
    otp: str
    new_password: str


@router.post("/register", response_model=UserOut, status_code=201)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user (free, open registration)."""
    result = await db.execute(select(User).where(User.username == payload.username))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already taken")

    result = await db.execute(select(User).where(User.email == payload.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


@router.post("/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    """Login with username or email + password. Returns JWT token."""
    # Try username first, then email
    result = await db.execute(select(User).where(User.username == form.username))
    user = result.scalar_one_or_none()

    if not user:
        result = await db.execute(select(User).where(User.email == form.username))
        user = result.scalar_one_or_none()

    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user.username})
    return Token(access_token=token)


@router.get("/me", response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user profile."""
    return current_user


# ── Forgot Password ───────────────────────────────────────────────────────────

@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Step 1 — Send a 6-digit OTP to the user's registered email."""
    from api.email_service import generate_otp, store_otp, send_otp_email

    result = await db.execute(select(User).where(User.email == payload.email.lower()))
    user = result.scalar_one_or_none()

    # Always return success to avoid email enumeration attacks
    if user:
        otp = generate_otp()
        store_otp(payload.email, otp)
        send_otp_email(payload.email, otp, username=user.username)

    return {"message": "If that email is registered, a reset code has been sent."}


@router.post("/verify-otp")
async def verify_otp_endpoint(payload: VerifyOTPRequest):
    """Step 2 — Verify the OTP is correct (without resetting yet)."""
    from api.email_service import verify_otp, store_otp, generate_otp

    # Peek — re-store the OTP temporarily so reset step can also verify it
    from api.email_service import _otp_store
    key = payload.email.lower()
    if key not in _otp_store:
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    stored_otp, expires_at = _otp_store[key]
    import time
    if time.time() > expires_at or stored_otp != payload.otp:
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    return {"message": "Code verified. You may now reset your password."}


@router.post("/reset-password")
async def reset_password(payload: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    """Step 3 — Verify OTP and set the new password."""
    from api.email_service import verify_otp

    if not verify_otp(payload.email, payload.otp):
        raise HTTPException(status_code=400, detail="Invalid or expired code")

    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    result = await db.execute(select(User).where(User.email == payload.email.lower()))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.hashed_password = hash_password(payload.new_password)
    await db.commit()

    return {"message": "Password reset successfully. You can now log in."}
