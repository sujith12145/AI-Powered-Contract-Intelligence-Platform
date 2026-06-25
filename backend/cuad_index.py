"""
CUAD Index Builder
==================
Reads master_clauses.csv once at startup and builds:
  1. A TF-IDF vectorizer + per-category centroid vectors for clause classification
  2. A ChromaDB in-process collection of all labeled clause texts for RAG

Everything is built in-memory; ChromaDB is persisted to .chroma_store/ so
subsequent restarts skip the embedding step.

Usage:
    from cuad_index import CUADIndex
    index = CUADIndex(csv_path="path/to/master_clauses.csv")
    index.build()   # idempotent — skips if already built
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# 41 CUAD category column names (excluding Document Name)
CUAD_CATEGORIES = [
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
    "Renewal Term",
    "Notice to Terminate Renewal",
    "Governing Law",
    "Most Favored Nation",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Customers",
    "Competitive Restriction Exception",
    "No-Solicit Of Employees",
    "Non-Disparagement",
    "Termination For Convenience",
    "Rofr/Rofo/Rofn",
    "Change Of Control",
    "Anti-Assignment",
    "Revenue/Profit Sharing",
    "Price Restrictions",
    "Minimum Commitment",
    "Volume Restriction",
    "Ip Ownership Assignment",
    "Joint Ip Ownership",
    "License Grant",
    "Non-Transferable License",
    "Affiliate License-Licensor",
    "Affiliate License-Licensee",
    "Unlimited/All-You-Can-Eat-License",
    "Irrevocable Or Perpetual License",
    "Source Code Escrow",
    "Post-Termination Services",
    "Audit Rights",
    "Uncapped Liability",
    "Cap On Liability",
    "Liquidated Damages",
    "Warranty Duration",
    "Insurance",
    "Covenant Not To Sue",
    "Third Party Beneficiary",
]

# Map CUAD category → friendly clause type shown in the UI
CATEGORY_TO_TYPE: Dict[str, str] = {
    "Parties": "Parties",
    "Agreement Date": "Term & Dates",
    "Effective Date": "Term & Dates",
    "Expiration Date": "Term & Dates",
    "Renewal Term": "Term & Dates",
    "Notice to Terminate Renewal": "Termination",
    "Governing Law": "Regulatory",
    "Most Favored Nation": "Pricing",
    "Non-Compete": "Regulatory",
    "Exclusivity": "Exclusivity",
    "No-Solicit Of Customers": "Regulatory",
    "Competitive Restriction Exception": "Regulatory",
    "No-Solicit Of Employees": "Regulatory",
    "Non-Disparagement": "Regulatory",
    "Termination For Convenience": "Termination",
    "Rofr/Rofo/Rofn": "IP Rights",
    "Change Of Control": "Termination",
    "Anti-Assignment": "Regulatory",
    "Revenue/Profit Sharing": "Payment Terms",
    "Price Restrictions": "Payment Terms",
    "Minimum Commitment": "Payment Terms",
    "Volume Restriction": "Payment Terms",
    "Ip Ownership Assignment": "IP Rights",
    "Joint Ip Ownership": "IP Rights",
    "License Grant": "IP Rights",
    "Non-Transferable License": "IP Rights",
    "Affiliate License-Licensor": "IP Rights",
    "Affiliate License-Licensee": "IP Rights",
    "Unlimited/All-You-Can-Eat-License": "IP Rights",
    "Irrevocable Or Perpetual License": "IP Rights",
    "Source Code Escrow": "IP Rights",
    "Post-Termination Services": "Termination",
    "Audit Rights": "Regulatory",
    "Uncapped Liability": "Liability",
    "Cap On Liability": "Liability",
    "Liquidated Damages": "Liability",
    "Warranty Duration": "Liability",
    "Insurance": "Liability",
    "Covenant Not To Sue": "IP Rights",
    "Third Party Beneficiary": "Regulatory",
    # Custom non-CUAD / boilerplate categories
    "Definitions": "General",
    "Confidentiality": "Confidentiality",
    "Force Majeure": "Regulatory",
    "General Provisions": "General",
    "Data Protection": "Data Protection",
    "Indemnification": "Indemnification",
    "SLAs": "SLAs",
    "General": "General",
}

# Risk weight per category — certain categories are inherently higher risk
CATEGORY_BASE_RISK: Dict[str, int] = {
    "Uncapped Liability": 30,
    "Cap On Liability": 20,
    "Ip Ownership Assignment": 25,
    "Joint Ip Ownership": 25,
    "Irrevocable Or Perpetual License": 20,
    "Non-Compete": 15,
    "Change Of Control": 15,
    "Anti-Assignment": 10,
    "Liquidated Damages": 15,
    "Revenue/Profit Sharing": 10,
    "Covenant Not To Sue": 20,
}


class CUADIndex:
    """
    Singleton-style index built from the CUAD master_clauses.csv.
    """

    def __init__(self, csv_path: str, chroma_persist_dir: str = ".chroma_store"):
        self.csv_path = Path(csv_path)
        self.chroma_persist_dir = chroma_persist_dir
        self.is_built = False

        # TF-IDF
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.category_centroids: Dict[str, np.ndarray] = {}
        self.category_exemplars: Dict[str, List[str]] = {}
        self.category_p90_distances: Dict[str, float] = {}

        # ChromaDB
        self._chroma_client: Optional[chromadb.Client] = None
        self._collection = None

        # Actual column names found in CSV (may differ slightly)
        self._col_map: Dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build(self) -> None:
        """Build the TF-IDF index and ChromaDB collection. Idempotent."""
        if self.is_built:
            return
        t0 = time.time()
        logger.info("Building CUAD index from %s …", self.csv_path)
        df = self._load_csv()
        self._build_tfidf(df)
        self._build_chroma(df)
        self.is_built = True
        logger.info("CUAD index ready in %.1fs", time.time() - t0)

    def classify_clause(self, text: str) -> Tuple[str, float]:
        """
        Returns (best_category, confidence_0_to_1).
        Uses rule-based checks first, then falls back to TF-IDF centroids.
        """
        if not self.is_built:
            return "General", 0.0

        first_line = text.split("\n")[0].lower()
        text_lower = text.lower()

        # Rule-based overrides for non-CUAD/boilerplate clauses
        if any(w in first_line for w in ["definition", "defined terms"]) or (len(text) < 1200 and "means the following" in text_lower):
            return "Definitions", 1.0
        if "force majeure" in first_line or "force majeure" in text_lower[:200]:
            return "Force Majeure", 1.0
        if any(w in first_line for w in ["governing law", "applicable law", "choice of law", "jurisdiction"]):
            return "Governing Law", 1.0
        if any(w in first_line for w in ["confidentiality", "non-disclosure", "nondisclosure", "proprietary info"]):
            return "Confidentiality", 1.0
        if any(w in first_line for w in ["miscellaneous", "general provisions", "entire agreement", "severability", "counterparts"]):
            return "General Provisions", 1.0
        if any(w in first_line for w in ["privacy", "data protection", "gdpr", "ccpa", "data security"]):
            return "Data Protection", 1.0
        if "indemni" in first_line or "indemni" in text_lower[:150]:
            return "Indemnification", 1.0
        if any(w in first_line for w in ["sla", "service level", "uptime", "support services"]):
            return "SLAs", 1.0

        # Heuristic rules for other CUAD categories to prioritize correct routing
        if any(w in first_line for w in ["liability", "limitation of liability", "indemnification and liability", "liability cap"]):
            if any(w in text_lower for w in ["uncapped", "unlimited", "no limit", "without limit"]):
                return "Uncapped Liability", 1.0
            return "Cap On Liability", 1.0
        if any(w in first_line for w in ["non-compete", "noncompete", "competition", "restrictive covenants"]):
            return "Non-Compete", 1.0
        if any(w in first_line for w in ["payment", "fees", "pricing", "billing", "invoice"]):
            return "Payment Terms", 1.0
        if any(w in first_line for w in ["termination", "terminate", "expiry", "survival"]):
            return "Termination For Convenience", 1.0
        if any(w in first_line for w in ["intellectual property", "ip rights", "ownership of", "work product"]):
            if "joint" in text_lower:
                return "Joint Ip Ownership", 1.0
            return "Ip Ownership Assignment", 1.0

        # Fall back to TF-IDF centroids
        vec = self.vectorizer.transform([text])
        best_cat, best_score = "General", 0.0
        for cat, centroid in self.category_centroids.items():
            score = float(cosine_similarity(vec, centroid.reshape(1, -1))[0, 0])
            if score > best_score:
                best_score = score
                best_cat = cat

        # Similarity threshold: if similarity is too low, classify as General
        if best_score < 0.18:
            return "General", best_score

        return best_cat, best_score

    def compute_risk_score(self, text: str, category: str) -> int:
        """
        Computes 0-100 risk score based on the risk rubric per category.
        """
        text_lower = text.lower()

        # 1. Definitions & General boilerplate
        if category in ("Definitions", "General Provisions", "General"):
            return 10

        # 2. Confidentiality
        if category == "Confidentiality":
            if any(w in text_lower for w in ["5 years", "five years", "10 years", "ten years", "perpetual"]):
                return 40  # medium
            if "receiving party" in text_lower and not "each party" in text_lower and not "mutual" in text_lower:
                return 45  # medium (one-sided)
            return 20  # low

        # 3. Force Majeure
        if category == "Force Majeure":
            if any(w in text_lower for w in ["supply chain", "shortage", "labor dispute"]):
                return 45  # medium (broad)
            return 20  # low

        # 4. Governing Law
        if category == "Governing Law":
            if any(w in text_lower for w in ["delaware", "new york", "california"]):
                return 18  # low
            if any(w in text_lower for w in ["england", "united kingdom", "foreign"]):
                return 35  # medium
            return 18  # low

        # 5. Data Protection
        if category == "Data Protection":
            # Check for data exploitation / machine learning training permissions (critical risk)
            if any(w in text_lower for w in ["machine learning", "train", "product improvement", "irrevocable license", "perpetual license"]):
                return 95  # critical
            if "reasonable security measures" in text_lower and not any(w in text_lower for w in ["iso", "27001", "soc", "encryption"]):
                return 45  # medium
            return 22  # low

        # 6. Indemnification
        if category in ("Indemnification", "Third Party Beneficiary"):
            # Check if one-sided against customer
            is_customer_indem = any(w in text_lower for w in ["customer shall indemnify", "buyer shall indemnify", "licensee shall indemnify"])
            is_vendor_indem = any(w in text_lower for w in ["vendor shall indemnify", "supplier shall indemnify", "licensor shall indemnify"])
            if is_customer_indem and not is_vendor_indem:
                return 88  # critical (one-sided)
            if is_customer_indem and is_vendor_indem or "mutual" in text_lower:
                return 45  # medium (mutual)
            if is_vendor_indem:
                return 20  # low (vendor indemnifies fully)
            return 45  # default medium

        # 7. Liability (Uncapped or Cap On Liability)
        if category in ("Uncapped Liability", "Cap On Liability"):
            if "uncapped" in text_lower or "unlimited" in text_lower or not any(w in text_lower for w in ["cap", "limit", "shall not exceed"]):
                return 95  # critical
            # Check for low cap
            if any(w in text_lower for w in ["one (1) month", "1 month", "three (3) month", "3 month", "30 days"]):
                return 85  # critical
            if any(w in text_lower for w in ["twelve (12) month", "12 month", "annual fees", "equal to the fees paid", "1x"]):
                return 45  # medium
            if any(w in text_lower for w in ["2x", "two times", "five times", "5x", "$1,000,000"]):
                return 22  # low
            return 45

        # 8. Termination
        if category == "Termination For Convenience":
            if any(w in text_lower for w in ["convenience", "any reason", "no reason"]):
                if any(w in text_lower for w in ["immediate", "5 days", "10 days", "30 days", "thirty (30) days"]):
                    return 72  # high (short notice)
            return 35  # medium

        # 9. Non-Compete
        if category == "Non-Compete":
            if any(w in text_lower for w in ["24 months", "two (2) years", "2 years", "two years"]):
                return 78  # high
            if any(w in text_lower for w in ["12 months", "one (1) year", "1 year"]):
                return 55  # medium
            return 60  # default medium

        # 10. IP Rights (Joint or Assignment)
        if category in ("Joint Ip Ownership", "Ip Ownership Assignment", "License Grant"):
            if "without accounting" in text_lower or "without consent" in text_lower:
                return 88  # critical
            if any(w in text_lower for w in ["whether or not during working hours", "all inventions conceived"]):
                return 55  # medium (overbroad assignment)
            if "assign" in text_lower or "sole property" in text_lower:
                return 35  # medium (standard assignment)
            return 22  # low (standard work-for-hire deliverables only)

        # 11. Payment Terms
        if category == "Payment Terms":
            if any(w in text_lower for w in ["1.5%", "18%", "12%", "suspend services"]):
                return 58  # medium (high interest/aggressive suspension)
            return 25  # low

        # 12. SLAs
        if category == "SLAs":
            if any(w in text_lower for w in ["99%", "98%", "credit cap of 20%", "20% cap"]):
                return 74  # high (poor uptime or low credits)
            if any(w in text_lower for w in ["99.9%", "99.99%"]):
                return 25  # low
            return 35  # default medium

        return 30  # fallback medium

    def get_exemplar(self, category: str) -> str:
        """Return the median-length exemplar clause text for a category."""
        examples = self.category_exemplars.get(category, [])
        if not examples:
            return "Standard market language for this clause type."
        # Pick the example closest to the median length
        lengths = [len(e) for e in examples]
        median_len = sorted(lengths)[len(lengths) // 2]
        closest = min(examples, key=lambda e: abs(len(e) - median_len))
        return closest[:1200]  # cap at 1200 chars for UI

    def query_rag(
        self, question: str, contract_id: Optional[str] = None, n: int = 3
    ) -> List[Dict]:
        """
        Query ChromaDB for clauses most relevant to `question`.
        If contract_id is provided, filter to only that contract's clauses.
        Returns list of {section, text, category, score}.
        """
        if self._collection is None:
            return []
        where = {"contract_id": contract_id} if contract_id else None
        try:
            results = self._collection.query(
                query_texts=[question],
                n_results=min(n, 10),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            out = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                out.append(
                    {
                        "section": meta.get("section", "§"),
                        "text": doc[:400],
                        "category": meta.get("category", ""),
                        "score": round(1.0 - dist, 3),
                    }
                )
            return out
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return []

    def index_contract_clauses(self, contract_id: str, clauses: List[Dict]) -> None:
        """
        Upsert a newly analyzed contract's clauses into ChromaDB so the
        Legal Assistant can answer questions about them.
        """
        if self._collection is None:
            return
        ids, docs, metas = [], [], []
        for clause in clauses:
            cid = f"{contract_id}__{clause['id']}"
            ids.append(cid)
            docs.append(clause["text"][:2000])
            metas.append(
                {
                    "contract_id": contract_id,
                    "section": clause.get("section", "§"),
                    "category": clause.get("cuadCategory", clause.get("type", "")),
                    "title": clause.get("title", ""),
                    "risk_level": clause.get("riskLevel", "low"),
                }
            )
        try:
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
        except Exception as exc:
            logger.warning("Failed to index clauses for %s: %s", contract_id, exc)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_csv(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path, dtype=str).fillna("")
        # Build flexible column map (case-insensitive, strip whitespace)
        col_lower = {c.strip().lower(): c for c in df.columns}
        for cat in CUAD_CATEGORIES:
            key = cat.lower()
            if key in col_lower:
                self._col_map[cat] = col_lower[key]
        logger.info("Loaded CSV with %d contracts, %d columns", len(df), len(df.columns))
        return df

    def _build_tfidf(self, df: pd.DataFrame) -> None:
        """Build TF-IDF vectorizer and per-category centroid vectors."""
        all_texts: List[str] = []
        cat_texts: Dict[str, List[str]] = {}

        for cat in CUAD_CATEGORIES:
            col = self._col_map.get(cat)
            if col is None:
                continue
            texts = [t.strip() for t in df[col].tolist() if len(t.strip()) > 30]
            if not texts:
                continue
            cat_texts[cat] = texts
            all_texts.extend(texts)
            self.category_exemplars[cat] = texts

        if not all_texts:
            logger.error("No clause texts found — check CSV column names")
            return

        logger.info("Fitting TF-IDF on %d clause texts …", len(all_texts))
        self.vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.vectorizer.fit(all_texts)

        # Compute centroids and p90 distances
        for cat, texts in cat_texts.items():
            vecs = self.vectorizer.transform(texts).toarray()
            centroid = vecs.mean(axis=0)
            self.category_centroids[cat] = centroid
            # Compute per-text distance from centroid
            sims = cosine_similarity(vecs, centroid.reshape(1, -1)).flatten()
            dists = 1.0 - sims
            p90 = float(np.percentile(dists, 90)) if len(dists) > 1 else 0.5
            self.category_p90_distances[cat] = max(p90, 0.05)

        logger.info("TF-IDF centroids built for %d categories", len(self.category_centroids))

    def _build_chroma(self, df: pd.DataFrame) -> None:
        """Build ChromaDB collection with all labeled CUAD clause texts."""
        os.makedirs(self.chroma_persist_dir, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(
            path=self.chroma_persist_dir,
        )

        try:
            col = self._chroma_client.get_collection("cuad_clauses")
            if col.count() > 0:
                self._collection = col
                logger.info("ChromaDB collection loaded from disk with %d documents", col.count())
                return
        except Exception:
            pass

        # Recreate collection to ensure fresh state
        try:
            self._chroma_client.delete_collection("cuad_clauses")
        except Exception:
            pass
        self._collection = self._chroma_client.create_collection(
            name="cuad_clauses",
            metadata={"hnsw:space": "cosine"},
        )

        ids, docs, metas = [], [], []
        doc_name_col = None
        for c in df.columns:
            if "document name" in c.lower():
                doc_name_col = c
                break

        for idx, row in df.iterrows():
            contract_name = (
                str(row[doc_name_col]).strip() if doc_name_col else f"contract_{idx}"
            )
            for cat in CUAD_CATEGORIES:
                col = self._col_map.get(cat)
                if col is None:
                    continue
                text = str(row[col]).strip()
                if len(text) < 30:
                    continue
                chunk_id = f"cuad_{idx}_{re.sub(r'\\W+', '_', cat)}"
                ids.append(chunk_id)
                docs.append(text[:2000])
                metas.append(
                    {
                        "contract_id": f"cuad_{idx}",
                        "contract_name": contract_name[:200],
                        "category": cat,
                        "section": cat,
                        "risk_level": "medium",
                        "title": cat,
                    }
                )

        # Upsert in batches of 500
        logger.info("Indexing %d clause chunks into ChromaDB …", len(ids))
        batch = 500
        for i in range(0, len(ids), batch):
            self._collection.add(
                ids=ids[i : i + batch],
                documents=docs[i : i + batch],
                metadatas=metas[i : i + batch],
            )
        logger.info("ChromaDB collection built with %d documents", len(ids))


# Global singleton — imported by main.py
_instance: Optional[CUADIndex] = None


def get_index() -> CUADIndex:
    global _instance
    if _instance is None:
        raise RuntimeError("CUADIndex not initialised — call init_index() first")
    return _instance


def init_index(csv_path: str, chroma_dir: str = ".chroma_store") -> CUADIndex:
    global _instance
    _instance = CUADIndex(csv_path=csv_path, chroma_persist_dir=chroma_dir)
    _instance.build()
    return _instance
