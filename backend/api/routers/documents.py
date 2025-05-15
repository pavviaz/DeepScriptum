from uuid import uuid4
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import JSONResponse
from celery import Celery
from celery.result import AsyncResult
import sqlalchemy as sa

from domain.auth.model import (
    BaseUser,
    RoleEnum,
    ShareRequest,
    ShareResponse,
    DocumentUpdateRequest,
)
from domain.exceptions import BaseAPIException
from infrastructure.postgres.models.document import DocumentDAO
from infrastructure.postgres.models.user import DocumentUserDAO, UserDAO
from api.dependencies import get_current_user
import settings
from api.utils import process_file


router = APIRouter(prefix="/documents", tags=["Documents"])

celery = Celery(__name__)
celery.conf.broker_url = settings.rabbitmq_settings.URI


@router.get("/")
async def get_all_docs(
    request: Request,
    user: str = Depends(get_current_user),
):
    _, pg_session = request.state.s3, request.state.db
    user_docs = {}

    q = (
        sa.select(DocumentUserDAO)
        .where(DocumentUserDAO.user_id == user)
        .order_by(DocumentUserDAO.last_access_at.asc())
    )
    q = await pg_session.execute(q)
    rows = q.scalars().all()

    if not rows:
        return user_docs

    for row in rows:
        q = sa.select(DocumentDAO).where(DocumentDAO.id == row.document_id)
        q = await pg_session.execute(q)
        res = q.fetchone()

        user_docs.update(
            {
                str(row.document_id): {
                    "name": res[0].name,
                    "s3_md_id": str(res[0].s3_md_id),
                }
            }
        )

    return JSONResponse(content=user_docs)


@router.post("/ocr")
async def create_ocr_task(
    request: Request,
    document: UploadFile = File(),
    user: str = Depends(get_current_user),
):
    s3_session, pg_session = request.state.s3, request.state.db

    doc_binary = document.file.read()
    document.file.seek(0)

    _ = process_file(doc_binary)  # check if file is pdf
    doc_s3_uuid = str(uuid4())

    try:
        await s3_session.upload_fileobj(
            document.file,
            settings.minio_settings.BUCKET,
            doc_s3_uuid,
            ExtraArgs={
                "Metadata": {
                    "ext": document.filename.split(".")[-1],
                },
            },
        )
    except:
        raise BaseAPIException(status_code=500, detail="S3 error")

    pg_raw_document = DocumentDAO(
        id=doc_s3_uuid,
        name=".".join(document.filename.split(".")[:-1]),
        s3_raw_id=doc_s3_uuid,
    )
    pg_session.add(pg_raw_document)
    await pg_session.commit()

    pg_doc_user = DocumentUserDAO(
        user_id=user, document_id=doc_s3_uuid, role=RoleEnum.owner
    )
    pg_session.add(pg_doc_user)
    await pg_session.commit()

    celery.send_task("images", task_id=doc_s3_uuid)

    return JSONResponse(
        status_code=201,
        content={
            "msg": "The task has been created successfully",
            "doc_id": doc_s3_uuid,
        },
    )


@router.get("/{document_id}/status")
async def check_doc_status(
    request: Request,
    document_id: str,
    user: BaseUser = Depends(get_current_user),
):
    _, pg_session = request.state.s3, request.state.db

    q = sa.select(DocumentDAO).where(DocumentDAO.id == document_id)
    q = await pg_session.execute(q)
    res = q.fetchone()[0]

    if res.s3_md_id:
        return {"s3_md_id": res.s3_md_id}

    return {"s3_md_id": "0"}


@router.get("/{document_db_id}/content")
async def get_document_content_by_db_id(
    request: Request,
    document_db_id: str,
    user_id: str = Depends(get_current_user),
):
    s3_session, pg_session = request.state.s3, request.state.db

    access_stmt = sa.select(DocumentUserDAO).where(
        DocumentUserDAO.user_id == user_id,
        DocumentUserDAO.document_id == document_db_id,
    )
    access_result = await pg_session.execute(access_stmt)
    if not access_result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Access denied to this document")

    doc_stmt = sa.select(DocumentDAO.s3_md_id, DocumentDAO.name).where(
        DocumentDAO.id == document_db_id
    )
    doc_info_result = await pg_session.execute(doc_stmt)
    doc_info = doc_info_result.one_or_none()

    if not doc_info or not doc_info.s3_md_id:
        raise HTTPException(
            status_code=404, detail="Document or its MD content not found"
        )

    s3_md_id_to_fetch = doc_info.s3_md_id
    document_name = doc_info.name

    try:
        response_s3 = await s3_session.get_object(
            Bucket=settings.minio_settings.BUCKET, Key=s3_md_id_to_fetch
        )
        async with response_s3["Body"] as stream:
            content = await stream.read()

        return {
            "content": content.decode("utf-8"),
            "name": document_name,
            "id": document_db_id,
            "s3_md_id": s3_md_id_to_fetch,
        }
    except Exception as e:
        request.state.Logger.error(f"S3 error fetching {s3_md_id_to_fetch}: {e}")
        raise HTTPException(
            status_code=500, detail="Could not fetch document content from storage"
        )


@router.put("/{s3_md_id}")
async def update_document_content(
    request: Request,
    s3_md_id: str,
    update_data: DocumentUpdateRequest,
    user_id: str = Depends(get_current_user),
):
    s3_session, pg_session = request.state.s3, request.state.db

    s3_key_to_update = None
    db_doc_id_for_check = None

    doc_check_stmt = sa.select(DocumentDAO.id).where(
        DocumentDAO.s3_md_id == s3_md_id
    )
    doc_check_res = await pg_session.execute(doc_check_stmt)
    db_doc_id_for_check = doc_check_res.scalar_one_or_none()
    if not db_doc_id_for_check:
        raise HTTPException(
            status_code=404,
            detail="Document not found by S3 MD ID",
        )
    s3_key_to_update = s3_md_id

    permission_stmt = sa.select(DocumentUserDAO).where(
        DocumentUserDAO.document_id == db_doc_id_for_check,
        DocumentUserDAO.user_id == user_id,
        DocumentUserDAO.role.in_([RoleEnum.owner, RoleEnum.editor]),
    )
    permission = (await pg_session.execute(permission_stmt)).scalar_one_or_none()
    if not permission:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to edit this document",
        )

    try:
        content_bytes = update_data.content.encode("utf-8")

        await s3_session.put_object(
            Bucket=settings.minio_settings.BUCKET,
            Key=s3_key_to_update,
            Body=content_bytes,
            ContentType="text/markdown; charset=utf-8",
        )

        stmt_update_access = (
            sa.update(DocumentUserDAO)
            .where(
                DocumentUserDAO.document_id == db_doc_id_for_check,
                DocumentUserDAO.user_id == user_id,
            )
            .values(last_access_at=datetime.utcnow())
        )
        await pg_session.execute(stmt_update_access)
        await pg_session.commit()

        return {"message": "Document updated successfully"}

    except Exception as e:
        request.state.Logger.error(
            f"Error updating document {s3_key_to_update} in S3: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Could not save document content",
        )


@router.get("/{document_id}")
async def get_document(
    request: Request,
    document_id: str,
    user: BaseUser = Depends(get_current_user),
):
    s3_session, _ = request.state.s3, request.state.db

    response = await s3_session.get_object(
        Bucket=settings.minio_settings.BUCKET, Key=document_id
    )
    async with response["Body"] as stream:
        content = await stream.read()

    return {"content": content.decode("utf-8")}


@router.post("/{document_id}/share")
async def share_document_with_user(
    request: Request,
    document_id: str,
    share_data: ShareRequest,
    current_user_id: str = Depends(get_current_user),
):
    pg_session = request.state.db

    doc_stmt = sa.select(DocumentDAO).where(DocumentDAO.id == document_id)
    doc_result = await pg_session.execute(doc_stmt)
    document = doc_result.scalar_one_or_none()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    permission_stmt = sa.select(DocumentUserDAO).where(
        DocumentUserDAO.document_id == document_id,
        DocumentUserDAO.user_id == current_user_id,
    )
    permission_result = await pg_session.execute(permission_stmt)
    user_permission = permission_result.scalar_one_or_none()

    if not user_permission or user_permission.role != RoleEnum.owner:
        raise HTTPException(
            status_code=403, detail="You do not have permission to share this document"
        )

    if share_data.role == RoleEnum.owner:
        raise HTTPException(
            status_code=400,
            detail="Cannot assign 'owner' role. Ownership is established at creation.",
        )

    target_user_stmt = sa.select(UserDAO).where(UserDAO.email == share_data.email)
    target_user_result = await pg_session.execute(target_user_stmt)
    target_user = target_user_result.scalar_one_or_none()

    if not target_user:
        raise HTTPException(
            status_code=404, detail=f"User with email '{share_data.email}' not found"
        )

    if str(target_user.id) == current_user_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot change your own role as an owner via this sharing method.",
        )

    existing_share_stmt = sa.select(DocumentUserDAO).where(
        DocumentUserDAO.document_id == document_id,
        DocumentUserDAO.user_id == target_user.id,
    )
    existing_share_result = await pg_session.execute(existing_share_stmt)
    existing_share_entry = existing_share_result.scalar_one_or_none()

    if existing_share_entry:
        if existing_share_entry.role == share_data.role:
            return ShareResponse(
                message=f"User {share_data.email} already has '{share_data.role.value}' access to this document.",
                user_email=share_data.email,
                role_assigned=share_data.role,
            )
        existing_share_entry.role = share_data.role
        existing_share_entry.last_access_at = datetime.utcnow()
        message = f"Access role for user {share_data.email} updated to '{share_data.role.value}'."
    else:
        new_share_entry = DocumentUserDAO(
            user_id=target_user.id,
            document_id=document_id,
            role=share_data.role,
            last_access_at=datetime.utcnow(),
        )
        pg_session.add(new_share_entry)
        message = f"Document shared with user {share_data.email} as '{share_data.role.value}'."

    await pg_session.commit()

    return ShareResponse(
        message=message, user_email=share_data.email, role_assigned=share_data.role
    )
