import logging
from contextlib import asynccontextmanager
from time import sleep

import aioboto3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

import settings
import api.healthchecker as hc
from infrastructure.postgres import database
from .routers import auth_router, doc_router


@asynccontextmanager
async def init_tables(app: FastAPI):
    urls_to_check = [f"{settings.minio_settings.URI}/minio/health/live"]

    hc.Readiness(urls=urls_to_check, logger=app.state.Logger).run()

    # Init DB tables if not exist
    for _ in range(5):
        try:
            async with database.engine.begin() as conn:
                await conn.run_sync(database.Base.metadata.create_all)
        except:
            print("DB is not ready yet")
            sleep(settings.app_settings.HC_SLEEP)

    # create bucket if not exist
    async with aioboto3.Session().client(
        "s3",
        endpoint_url=settings.minio_settings.URI,
        aws_access_key_id=settings.minio_settings.ROOT_USER,
        aws_secret_access_key=settings.minio_settings.ROOT_PASSWORD,
    ) as s3_client:
        try:
            await s3_client.head_bucket(Bucket=settings.minio_settings.BUCKET)
        except Exception:
            await s3_client.create_bucket(Bucket=settings.minio_settings.BUCKET)

    yield


app = FastAPI(
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=init_tables,
)

app.state.Logger = logging.getLogger(name="deepscriptum_backend")
app.state.Logger.setLevel("DEBUG")

app.include_router(auth_router, prefix="/api")
app.include_router(doc_router, prefix="/api")


@app.get("/")
async def root():
    return {"info": "DeepScriptum API. See /docs for documentation"}


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    request.state.db = database.async_session_factory()
    async with aioboto3.Session().client(
        "s3",
        endpoint_url=settings.minio_settings.URI,
        aws_access_key_id=settings.minio_settings.ROOT_USER,
        aws_secret_access_key=settings.minio_settings.ROOT_PASSWORD,
    ) as s3_client:
        request.state.s3 = s3_client
        try:
            response = await call_next(request)
        except Exception as exc:
            print(exc, flush=True)
            detail = getattr(exc, "detail", None)
            unexpected_error = not detail
            if unexpected_error:
                args = getattr(exc, "args", None)
                detail = args[0] if args else str(exc)
            status_code = getattr(exc, "status_code", 500)
            response = JSONResponse(
                content={"detail": str(detail), "success": False},
                status_code=status_code,
            )
        finally:
            await request.state.db.close()

    return response


origins = [
    "http://frontend",
    f"http://frontend:{settings.front_settings.PORT}",
    "http://localhost",
    f"http://localhost:{settings.front_settings.PORT}",
    f"http://127.0.0.1:{settings.front_settings.PORT}",
    f"http://0.0.0.0:{settings.front_settings.PORT}",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
