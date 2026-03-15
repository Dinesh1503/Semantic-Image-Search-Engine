from fastapi import FastAPI,WebSocket, WebSocketDisconnect, Request
from inference import Inference
from contextlib import asynccontextmanager
import os,asyncio
from dotenv import load_dotenv
from database.db import VectorDatabase
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

class SearchRequest(BaseModel):
    caption: str

@asynccontextmanager
async def lifespan(app:FastAPI):

    load_dotenv()
    
    app.state.inf = Inference()


    db_name = os.getenv("DB")
    user = os.getenv("DB_USER")
    port = os.getenv("PORT")
    password = os.getenv("PASSWORD")

    app.state.db = VectorDatabase(db_name,password,user,port)

    app.state.db.connect()

    yield 

    await app.state.db.disconnect()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/images", StaticFiles(directory="/Users/dinesh/Project/Semantic Image Search Engine/engine/test_data/a"), name="images")

@app.websocket("/ws")
async def websocket_connection(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        asyncio.get_event_loop().stop()



@app.post("/search")
async def call_backend(request: Request, body: SearchRequest):

    user_query_caption = body.caption
    vectorised = app.state.inf.vectorise_query(user_query_caption)
    print("\n vectorsied: ",vectorised)

    results = app.state.db.retrieve_data(vectorised)
    print("\n resutls: ",results)

    classified_results = app.state.inf.classify_results(results)
    print(classified_results)

    return classified_results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    


    
    




