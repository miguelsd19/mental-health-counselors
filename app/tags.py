# app/tags_en.py
import re
import unicodedata
from typing import List, Iterable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text

def _tokenizer(text: str):
    text = _normalize(text)
    return re.findall(r"[a-z]{3,}", text)

def build_vectorizer(max_features: int = 30000) -> TfidfVectorizer:
    return TfidfVectorizer(
        tokenizer=_tokenizer,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        stop_words="english",
    )

def extract_keywords_batch(
    docs: Iterable[str],
    top_k: int = 5,
    vectorizer: TfidfVectorizer | None = None,
) -> List[List[str]]:
    docs = list(docs)
    if not docs:
        return []

    vec = vectorizer or build_vectorizer()
    X = vec.fit_transform(docs) 
    vocab = np.array(vec.get_feature_names_out())

    tags_per_doc: List[List[str]] = []
    for i in range(X.shape[0]):
        row = X.getrow(i)
        if row.nnz == 0:
            tags_per_doc.append([])
            continue

        idxs = row.toarray().ravel().argsort()[::-1]
        chosen = []
        roots = set()
        for j in idxs:
            term = vocab[j]
            root = term.split()[0]
            if root in roots:
                continue
            chosen.append(term)
            roots.add(root)
            if len(chosen) >= top_k:
                break
        tags_per_doc.append(chosen)
    return tags_per_doc
