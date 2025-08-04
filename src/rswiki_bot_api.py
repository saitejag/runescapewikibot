from fastapi import FastAPI
from pydantic import BaseModel
# try:
#     from .rswiki_bot import RswikiBot
# except:
#     from rswiki_bot import RswikiBot
from rswiki_bot import RswikiBot
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

class querySchema(BaseModel):
    query:str

app = FastAPI()
bot = RswikiBot()
@app.post('/api/query',response_model=str)
async def post_query(q:querySchema):
    result = await bot.ask_llm(q.query)
    return result["answer"]

@app.get("/")
async def read_root():
    if os.path.exists("dist/index.html"):
        from fastapi.responses import FileResponse
        return FileResponse('dist/index.html')
    return {"message": "Phonebook API is running"}

if os.path.exists("dist"):
    app.mount("/static", StaticFiles(directory="dist",html=False))


def run():
    uvicorn.run("rswiki_bot_api:app", host="127.0.0.1", port=8000, reload=True)
