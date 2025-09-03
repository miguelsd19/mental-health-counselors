import json
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from openai import OpenAI
import pandas as pd
from io import StringIO


from .db import Base, engine, SessionLocal
from .schemas import IngestItem, AskRequest, AskResponse, Retrieved
from .ingest import ingest_items
from .rag import search, rebuild_faiss, SYSTEM_PROMPT_ES, USER_PROMPT_TEMPLATE_ES
from .settings import OPENAI_API_KEY, OPENAI_MODEL


app = FastAPI(title="Simple API", version="1.0.0")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/search")
async def search_preview(q: str, k: int = 5, db: Session = Depends(get_db)):
    rows = search(db, q, top_k=k)
    out = []
    for score, assoc in rows:
        out.append({
        "id": assoc.id,
        "score": score,
        "question": assoc.question.text,
        "answer": assoc.answer.text,
        "source": assoc.source,
        "tags": assoc.tags.split(",") if assoc.tags else None,
        })
    return out


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, db: Session = Depends(get_db)):
    rows = search(db, req.query, top_k=req.top_k)
    if not rows:
        retrieved = []
        context_str = "(No relevant context)"
    else:
        retrieved = [
            Retrieved(
            id=assoc.id,
            score=score,
            question=assoc.question.text,
            answer=assoc.answer.text,
            source=assoc.source,
            tags=assoc.tags.split(",") if assoc.tags else None,
            )
            for (score, assoc) in rows
        ]
    
    parts = []
    for r in retrieved:
        parts.append(f"- Q: {r.question}\n A: {r.answer}\n Fuente: {r.source or 'N/A'}\n")
    context_str = "\n".join(parts)


    if not OPENAI_API_KEY:
        suggestion = (
        "No OpenAI key"
        )
        return AskResponse(suggestion=suggestion, retrieved=retrieved)


    client = OpenAI(api_key=OPENAI_API_KEY)


    messages = [
    {"role": "system", "content": SYSTEM_PROMPT_ES},
    {"role": "user", "content": USER_PROMPT_TEMPLATE_ES.format(query=req.query, context=context_str)},
    ]


    try:
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages, temperature=0.3)
        suggestion = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")


    return AskResponse(suggestion=suggestion, retrieved=retrieved)


@app.post("/ingest-csv")
async def ingest_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="A.csv is expected with clomuns Context,Response")
    content = (await file.read()).decode("utf-8", errors="ignore")
    df = pd.read_csv(StringIO(content))
    
    cols = {c.strip(): c for c in df.columns}
    if "Context" not in cols or "Response" not in cols:
        raise HTTPException(status_code=400, detail=f"Comuns not found, found: {list(df.columns)}")

    df = df.rename(columns={cols["Context"]: "Context", cols["Response"]: "Response"})
    df = df[["Context", "Response"]].dropna().drop_duplicates(subset=["Context", "Response"]).reset_index(drop=True)

    payload = [
        IngestItem(
            question=str(row.Context).strip(),
            answer=str(row.Response).strip(),
            source="kaggle:train",
            tags=None
        )
        for _, row in df.iterrows()
    ]

    count = ingest_items(db, payload)
    return {"ingested": count}