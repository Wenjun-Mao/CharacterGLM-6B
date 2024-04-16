import json
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI()

MODEL_PATH = "./CharacterGLM-6B"
TOKENIZER_PATH = MODEL_PATH

# with open("character.json", "r", encoding="utf-8") as file:
#     characters = json.load(file)

url = "https://xbuibg.cn/fx/uploads/bacimage/renwu.json"
try:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
except requests.RequestException as e:
    raise HTTPException(status_code=500, detail=str(e))

characters = {}
for item in response.json():
    name = item["name"]
    characters[name] = {
        "user_info": "",
        "bot_info": item["jianjie"],
        "user_name": "用户",
        "bot_name": name,
        "greeting": item["zhaohu"],
    }


class ChatRequest(BaseModel):
    character_choice: str
    query: str
    history: list = []
    max_length: int = 2048
    top_p: float = 0.8
    temperature: float = 0.9
    repetition_penalty: float = 1.0
    num_beams: int = 1


class ChatResponse(BaseModel):
    response: str
    history: list


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True, device_map="auto"
).eval()


@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.character_choice not in characters:
        raise HTTPException(status_code=404, detail="Character not found")

    character_data = characters[request.character_choice]
    session_meta = {
        "user_info": character_data.get("user_info", ""),
        "bot_info": character_data.get("bot_info", ""),
        "bot_name": character_data.get("bot_name", ""),
        "user_name": character_data.get("user_name", "user"),
    }
    greeting = character_data.get("greeting", "")
    if not request.history:
        if greeting:
            request.history.append(("", greeting))

    history = request.history

    response, history = model.chat(
        tokenizer,
        session_meta=session_meta,
        query=request.query,
        history=history,
        max_length=request.max_length,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        num_beams=request.num_beams,
    )

    return ChatResponse(response=response, history=history)
