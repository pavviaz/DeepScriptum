import io
from multiprocessing.pool import ThreadPool
from typing import List
from uuid import uuid4

from celery import Celery
from openai import OpenAI
import boto3
from sqlalchemy import update
import httpx

import config
import settings
from utils import process_file
from infrastructure.postgres import database
from infrastructure.postgres.models import DocumentDAO


REPLACERS = {"latex": config.LATEX_REPLACER, "md": config.MD_REPLACER}


client = OpenAI(
    api_key=settings.app_settings.OPENAI_KEY,
    http_client=httpx.Client(proxy=settings.app_settings.PROXY),
)

celery = Celery(__name__)
celery.conf.broker_url = settings.rabbitmq_settings.URI
celery.conf.result_backend = settings.redis_settings.URI


@celery.task(
    name="images", bind=True, time_limit=600, soft_time_limit=540, track_started=True
)
def process_images(self, decode_type: str = "md"):
    task_id = self.request.id

    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.minio_settings.URI,
        aws_access_key_id=settings.minio_settings.ROOT_USER,
        aws_secret_access_key=settings.minio_settings.ROOT_PASSWORD,
    )

    response = s3_client.get_object(Bucket=settings.minio_settings.BUCKET, Key=task_id)
    data = response["Body"].read()
    imgs_binary = process_file(data, file_type=response["Metadata"]["ext"])

    replacer = REPLACERS[decode_type]
    imgs_enum = [(idx, img, replacer) for idx, img in enumerate(imgs_binary)]

    try:
        with ThreadPool(len(imgs_enum)) as thread_pool:
            map_results = thread_pool.starmap(_apply_map_ocr, imgs_enum)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)

    reduce_result = _apply_reduce_ocr(map_results, decode_type, replacer)

    doc_s3_uuid = str(uuid4())
    try:
        s3_client.upload_fileobj(
            io.BytesIO(reduce_result.encode("utf-8")),
            settings.minio_settings.BUCKET,
            doc_s3_uuid,
            ExtraArgs={"ContentType": "text/markdown"},
        )
    except:
        raise self.retry(exc=exc, countdown=5)
        
    pg_session = database.session_factory()
    stmt = (
        update(DocumentDAO)
        .where(DocumentDAO.id == task_id)
        .values(s3_md_id=doc_s3_uuid)
    )
    pg_session.execute(stmt)
    pg_session.commit()

    return doc_s3_uuid


def _apply_map_ocr(idx: int, img_binary: str, replacer: tuple):
    payload = {
        "model": settings.app_settings.MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": config.DEFAULT_MAP_PROMPT.format(
                            idx + 1, replacer[0], replacer[0], replacer[0]
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_binary}"},
                    },
                ],
            }
        ],
    }

    try:
        response = client.chat.completions.create(
            model=settings.app_settings.MODEL_NAME,
            messages=payload["messages"],
        )
        content = response.choices[0].message.content
        content = content.lstrip(replacer[1]).rstrip(replacer[2]).strip("\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return content


def _apply_reduce_ocr(texts: List[str], decode_type: str, replacer: tuple):
    payload = {
        "model": settings.app_settings.MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": config.DEFAULT_REDUCE_PROMPT.format(
                            len(texts),
                            decode_type,
                            config.REDUCE_SPLITTER,
                            decode_type,
                            f"\n{config.REDUCE_SPLITTER}\n".join(texts),
                        ),
                    }
                ],
            }
        ],
    }

    try:
        response = client.chat.completions.create(
            model=settings.app_settings.MODEL_NAME,
            messages=payload["messages"],
        )
        content = response.choices[0].message.content
        content = content.lstrip(replacer[1]).rstrip(replacer[2]).strip("\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return content


@celery.task(
    name="texts", bind=True, time_limit=600, soft_time_limit=540, track_started=True
)
def process_texts(self, decode_type: str = "md"):
    task_id = self.request.id

    replacer = REPLACERS[decode_type]
    if texts == ["unknown"]:
        return "Unknown data type"
    texts_enum = [(idx, text, replacer) for idx, text in enumerate(texts)]

    try:
        with ThreadPool(len(texts_enum)) as thread_pool:
            map_results = thread_pool.starmap(_apply_map_text, texts_enum)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)

    reduce_result = _apply_reduce_text(map_results, decode_type, replacer)
    return reduce_result


def _apply_map_text(idx: int, text: str, replacer: tuple):
    payload = {
        "model": settings.app_settings.MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": config.DEFAULT_MAP_PROMPT.format(
                            idx + 1, replacer[0], replacer[0], replacer[0]
                        ),
                    },
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            }
        ],
    }

    try:
        response = client.chat.completions.create(
            model=settings.app_settings.MODEL_NAME,
            messages=payload["messages"],
        )
        content = response.choices[0].message.content
        content = content.lstrip(replacer[1]).rstrip(replacer[2]).strip("\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return content


def _apply_reduce_text(texts: List[str], decode_type: str, replacer: tuple):
    payload = {
        "model": settings.app_settings.MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": config.DEFAULT_REDUCE_PROMPT.format(
                            len(texts),
                            decode_type,
                            config.REDUCE_SPLITTER,
                            decode_type,
                            f"\n{config.REDUCE_SPLITTER}\n".join(texts),
                        ),
                    }
                ],
            }
        ],
    }

    try:
        response = client.chat.completions.create(
            model=settings.app_settings.MODEL_NAME,
            messages=payload["messages"],
        )
        content = response.choices[0].message.content
        content = content.lstrip(replacer[1]).rstrip(replacer[2]).strip("\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return content
