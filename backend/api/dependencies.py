from fastapi import Depends, Request, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from infrastructure.postgres.service.auth.repo import UserAuthRepository
from domain.exceptions import InvalidTokenError


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="fake_url")


def get_db(request: Request):
    return request.state.db


def get_auth_repository(
    session: AsyncSession = Depends(get_db),
):
    return UserAuthRepository(session)


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        # print(f"Received token for verification: {token}")
        user_id = UserAuthRepository.verify_token(token)
        if not user_id:
            # print("Token verification returned no user_id.")
            raise InvalidTokenError(detail="Invalid token or user not found")
        # print(f"Token verified. User ID: {user_id}")
        return user_id
    except InvalidTokenError as e:
        print(f"InvalidTokenError: {e.detail}")
        raise HTTPException(
            status_code=401,
            detail=e.detail or "Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (
        JWTError,
        ValidationError,
    ) as e_jwt_val:
        print(f"JWTError/ValidationError: {str(e_jwt_val)}")
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials (JWT/Validation Error)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e_generic:
        print(f"Generic exception during token verification: {str(e_generic)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during token verification.",
            headers={"WWW-Authenticate": "Bearer"},
        )
