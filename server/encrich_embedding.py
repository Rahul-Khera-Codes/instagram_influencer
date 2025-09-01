#!/usr/bin/env python3
"""Batch enrichment script: compute embeddings for all influencers missing them.

Usage:
    python scripts/enrich_embeddings.py

It loads `MONGO_URI` from environment and uses the same `get_embedding` implementation
in `server/influencers.py` by importing it. This script intentionally keeps the Mongo
update simple (per-document update) and sleeps briefly to avoid rate limits.
"""
import os
import time
from dotenv import load_dotenv

load_dotenv()

from server.influencers import get_embedding
from server.db import client

if not client:
    print("MongoDB client not configured via MONGO_URI. Set MONGO_URI in .env and try again.")
    raise SystemExit(1)

db = client["influencer_db"]
influencers = db["influencers"]

def enrich_all(batch_size: int = 50, delay: float = 0.25):
    cursor = influencers.find({"embedding": {"$exists": False}}).limit(batch_size)
    count = 0
    for doc in cursor:
        text = " ".join(filter(None, [doc.get("username", ""), doc.get("bio", ""), doc.get("category", "")]))
        try:
            emb = get_embedding(text)
        except Exception as e:
            print(f"failed to get embedding for {doc.get('username') or doc.get('_id')}: {e}")
            continue
        influencers.update_one({"_id": doc["_id"]}, {"$set": {"embedding": emb}})
        count += 1
        print(f"updated {doc.get('username') or doc.get('_id')} (#{count})")
        time.sleep(delay)
    print(f"done — updated {count} documents")

if __name__ == '__main__':
    enrich_all()


