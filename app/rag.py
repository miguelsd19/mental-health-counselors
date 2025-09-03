import os
import json
import re
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session


from .models import Question, Answer, QAAssociation
from .settings import EMBEDDING_MODEL, FAISS_DIR, TOP_K, MIN_SCORE

PHI_REGEX = re.compile(r"\b([A-Z][a-z]+\s[A-Z][a-z]+|\d{2,})\b")


def clean_text(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = PHI_REGEX.sub("[REDACTED]", t)
    return t

# --- Embeddings/FAISS ---
_model = None
_index = None
_mapping_path = Path(FAISS_DIR) / "mapping.json"
_index_path = Path(FAISS_DIR) / "index.faiss"




def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def ensure_index_dir():
    Path(FAISS_DIR).mkdir(parents=True, exist_ok=True)



def load_index() -> Tuple[faiss.IndexFlatIP, list]:
    """Return (index, mapping_list) where mapping_list[i] = association_id"""
    ensure_index_dir()
    mapping = []
    if _index_path.exists() and _mapping_path.exists():
        index = faiss.read_index(str(_index_path))
        mapping = json.loads(Path(_mapping_path).read_text())
    else:
        index = faiss.IndexFlatIP(384) # all-MiniLM-L6-v2 dim
    return index, mapping


def save_index(index: faiss.IndexFlatIP, mapping: list):
    ensure_index_dir()
    faiss.write_index(index, str(_index_path))
    Path(_mapping_path).write_text(json.dumps(mapping))


def embed_texts(texts: List[str]) -> np.ndarray:
    model = get_model()
    emb = model.encode(texts, normalize_embeddings=True)
    return np.asarray(emb, dtype="float32")



def upsert_qa(db: Session, question: str, answer: str, source: str = None, tags: List[str] = None) -> int:
    q = Question(text=clean_text(question))
    a = Answer(text=clean_text(answer))
    db.add_all([q, a])
    db.flush()
    assoc = QAAssociation(question_id=q.id, answer_id=a.id, source=source, tags=",".join(tags) if tags else None)
    db.add(assoc)
    db.commit()
    return assoc.id




def rebuild_faiss(db: Session):
    index = faiss.IndexFlatIP(384)
    mapping = []
    items = (
    db.query(QAAssociation.id, Question.text, Answer.text)
    .join(Question, QAAssociation.question_id == Question.id)
    .join(Answer, QAAssociation.answer_id == Answer.id)
    .all()
    )
    texts = [f"Q: {clean_text(q)}\nA: {clean_text(a)}" for (_id, q, a) in items]
    
    if not texts:
        save_index(index, mapping)
        return
    
    embs = embed_texts(texts)
    index.add(embs)
    mapping = [int(_id) for (_id, _q, _a) in items]
    save_index(index, mapping)
    



def search(db: Session, query: str, top_k: int = None, min_score: float = MIN_SCORE):
    top_k = top_k or TOP_K
    index, mapping = load_index()
    
    if index.ntotal == 0:
        return []
    qvec = embed_texts([clean_text(query)])
    scores, idxs = index.search(qvec, top_k)
    results = []
    
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        s = float(score)

        if s < min_score:
            continue
        assoc_id = mapping[idx]
        assoc = db.query(QAAssociation).get(assoc_id)
        results.append((s, assoc))
    return results


SYSTEM_PROMPT_ES = (
"You are a clinical assistant for therapists. Offer suggestions based on the context, "
"demonstrate empathy, avoid formal diagnoses or medical advice. "
"Include practical strategies (e.g., brief psychoeducation, open-ended questions, grounding techniques). "
"If the situation suggests risk, recommend safety protocols and referral. "
"Do not assume emergencies; if there are clear signs, suggest contacting emergency services."
)


USER_PROMPT_TEMPLATE_ES = (
"Therapist's question: \n{query}\n\n"
"Retrieved context (similar cases and therapists' responses):\n{context}\n\n"
"Based on the context, draft a brief suggestion (<=200 words), structured as: "
"1) Validation/empathy, 2) Brief conceptual framework, 3) Actions/strategies in session and between sessions, 4) Note on scope."
)