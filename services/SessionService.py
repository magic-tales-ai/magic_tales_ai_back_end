import jwt
import datetime
import os
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="session/login-swagger")


async def check_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(
            token, os.getenv("JWT_SECRET_KEY"), algorithms=[os.getenv("JWT_ALGORITHM")]
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def refresh_access_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(
            token, os.getenv("JWT_SECRET_KEY"), algorithms=[os.getenv("JWT_ALGORITHM")]
        )  # decode method includes validation
        payload["exp"] = datetime.datetime.now(
            tz=datetime.timezone.utc
        ) + datetime.timedelta(minutes=int(os.getenv("JWT_EXP_TIME_IN_MINUTES")))
        encoded_jwt = jwt.encode(
            payload, os.getenv("JWT_SECRET_KEY"), algorithm=os.getenv("JWT_ALGORITHM")
        )
        return encoded_jwt
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
