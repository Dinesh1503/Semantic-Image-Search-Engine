from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException, status
from inference import Inference
from contextlib import asynccontextmanager
import os
import secrets
from dotenv import load_dotenv
from database.db import VectorDatabase
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from paths import images_dir

MAX_QUERY_LENGTH = 500
DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173"


class SearchRequest(BaseModel):
    model_config = {"extra": "forbid"}

    caption: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)


def get_allowed_origins() -> list[str]:
    """Browser origins allowed to call the API (override with ALLOWED_ORIGINS)."""
    load_dotenv()
    raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def require_api_key(request: Request) -> None:
    """Requires a matching X-API-Key header when API_KEY is configured."""
    expected = os.getenv("API_KEY")
    if not expected:
        return
    provided = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.inf = Inference()

    app.state.db = VectorDatabase.from_env()

    app.state.db.connect()

    yield

    app.state.db.disconnect()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

served_images_dir = images_dir()
if not served_images_dir.is_dir():
    raise RuntimeError(f"IMAGES_PATH is not a directory: {served_images_dir}")

app.mount("/images", StaticFiles(directory=served_images_dir), name="images")


@app.websocket("/ws")
async def websocket_connection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        return


@app.post("/search", dependencies=[Depends(require_api_key)])
async def call_backend(body: SearchRequest):

    vectorised = app.state.inf.vectorise_query(body.caption)
    results = app.state.db.retrieve_data(vectorised)

    return app.state.inf.classify_results(results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("SERVER_PORT", "8000")))
