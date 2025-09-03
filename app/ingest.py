from typing import Iterable
from sqlalchemy.orm import Session
from .schemas import IngestItem
from .rag import upsert_qa, rebuild_faiss
from .tags import extract_keywords_batch


def ingest_items(db: Session, items: Iterable[IngestItem], auto_tags: bool = True, top_k: int = 5):
    items = list(items)
    if auto_tags:
        docs, idx_needs = [], []
        for i, it in enumerate(items):
            if not it.tags:
                docs.append(f"{it.question}\n{it.answer}")
                idx_needs.append(i)
        if docs:
            tags_lists = extract_keywords_batch(docs, top_k=top_k)
            for pos, tags in zip(idx_needs, tags_lists):
                items[pos].tags = tags or None

    count = 0
    for it in items:
        upsert_qa(db, it.question, it.answer, it.source, it.tags)
        count += 1
    rebuild_faiss(db)
    return count