from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
import jwt
from datetime import datetime, timedelta, timezone
import os

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="session/login-swagger")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your_default_secret_key")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_TIME_IN_MINUTES = int(os.getenv("JWT_EXP_TIME_IN_MINUTES", "30"))


def decode_jwt(token: str):
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail="Invalid token") from e


async def check_token(token: str = Depends(oauth2_scheme)) -> dict:
    """Asynchronously validates the JWT token."""
    return decode_jwt(token)


async def refresh_access_token(token: str = Depends(oauth2_scheme)) -> str:
    """Asynchronously refreshes the JWT access token."""
    payload = decode_jwt(token)
    payload["exp"] = datetime.now(tz=timezone.utc) + timedelta(
        minutes=JWT_EXP_TIME_IN_MINUTES
    )
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
