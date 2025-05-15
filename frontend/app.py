import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import healthchecker as hc
import settings


@asynccontextmanager
async def init_tables(app: FastAPI):
    urls_to_check = [settings.back_settings.URI]

    hc.Readiness(urls=urls_to_check, logger=app.state.Logger).run()

    yield


app = FastAPI(
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=init_tables,
)

app.state.Logger = logging.getLogger(name="deepscriptum_backend")
app.state.Logger.setLevel("DEBUG")

templates = Jinja2Templates("templates")
app.mount("/assets", StaticFiles(directory="templates/assets"), name="assets")


@app.get("/")
async def start_page(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)


@app.get("/auth/login")
async def login_page(request: Request):
    return templates.TemplateResponse(
        name="login.html",
        context={
            "request": request,
            "LOGIN_ENDPOINT": "http://0.0.0.0:8080/api/auth/login",
        },
    )


@app.get("/auth/register")
async def register_page(request: Request):
    return templates.TemplateResponse(
        name="signup.html",
        context={
            "request": request,
            "REGISTER_ENDPOINT": "http://0.0.0.0:8080/api/auth/register",
        },
    )


@app.get("/home")
async def home_page(request: Request):
    return templates.TemplateResponse(name="home.html", request=request)


@app.get("/editor/{document_id}")
async def editor_page(request: Request):
    return templates.TemplateResponse(name="dashboard.html", request=request)
