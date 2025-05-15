import enum
from datetime import datetime

from pydantic import BaseModel, Field, EmailStr


class RoleEnum(enum.Enum):
    owner = "owner"
    viewer = "viewer"
    editor = "editor"


class ShareRequest(BaseModel):
    email: EmailStr
    role: RoleEnum


class DocumentUpdateRequest(BaseModel):
    content: str


class ShareResponse(BaseModel):
    message: str
    user_email: EmailStr
    role_assigned: RoleEnum


class BaseUser(BaseModel):
    email: str

    class Config:
        from_attributes = True


class UserAuth(BaseUser):
    password: str = Field(min_length=4)


class JWTPayload(BaseModel):
    user: BaseUser
    sub: str
    iat: datetime
    exp: datetime


class Token(BaseModel):
    access_token: str
