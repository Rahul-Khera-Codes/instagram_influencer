from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import requests
import os
from dotenv import load_dotenv
from math import log10
import re
import json
from db import searches_collection, influencers_collection, HAS_TEXT_INDEX
from datetime import datetime
import time
import random
from functools import lru_cache
from fastapi import Depends
from auth import get_current_user
import numpy as np

load_dotenv()

router = APIRouter()

RAPIDAPI_HOST = "instagram-best-experience.p.rapidapi.com"
RAPIDAPI_BASE = f"https://{RAPIDAPI_HOST}"
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

OPENAI_KEY = os.getenv("OPENAI_KEY")




def get_embedding(text: str) -> list[float]:
    """Generate OpenAI embedding for text (bio, username, etc.)."""
    if not text:
        return []
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    body = {
        "input": text,
        "model": "text-embedding-3-small"   # cheaper & good for semantic search
    }
    resp = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=body, timeout=15)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI embedding error: {resp.text}")
    data = resp.json()
    return data["data"][0]["embedding"]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    av = np.array(a, dtype=float)
    bv = np.array(b, dtype=float)
    denom = (np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _merge_results_with_semantic(results: list[Dict[str, Any]], query: str, limit: int) -> tuple[list[Dict[str, Any]], list[str]]:
    """Call semantic search and merge unique semantic hits into `results` up to `limit`.

    Returns (merged_results, expanded_terms)
    """
    merged = list(results or [])
    expanded_terms: list[str] = []
    try:
        sem = semantic_search(query=query, top_k=limit, expand=True)
        sem_results = sem.get("results", []) if isinstance(sem, dict) else []
        expanded_terms = sem.get("expanded_terms", []) if isinstance(sem, dict) else []
    except Exception:
        sem_results = []

    existing_usernames = set((r.get("username") or "").lower() for r in merged)
    for s in sem_results:
        uname = (s.get("username") or "").lower()
        if not uname or uname in existing_usernames:
            continue
        # normalize semantic result shape to match RapidAPI profile shape where possible
        merged.append({
            "pk": s.get("pk") or None,
            "username": s.get("username"),
            "full_name": s.get("full_name") or s.get("name") or None,
            "followers": s.get("followers") or s.get("follower_count") or None,
            "profile_pic": s.get("profile_pic") or s.get("profile_pic_url") or None,
            "bio": s.get("bio") or s.get("biography") or None,
            "semantic_score": s.get("score") if s.get("score") is not None else None,
        })
        existing_usernames.add(uname)
        if len(merged) >= int(limit):
            break

    return merged, expanded_terms


@router.post("/enrich/update_embedding")
def enrich_influencer_embedding(user_id: str):
    """Generate and store embedding for a single influencer document identified by user_id (pk).

    This endpoint is best-effort and intended for manual/enrichment scripts.
    """
    if influencers_collection is None:
        raise HTTPException(status_code=500, detail="MongoDB influencers collection not configured")

    # find influencer by user_id (could be stored as 'pk' or 'user_id')
    doc = influencers_collection.find_one({"$or": [{"pk": user_id}, {"user_id": user_id}, {"_id": user_id}]})
    if not doc:
        raise HTTPException(status_code=404, detail="influencer not found")

    text = " ".join(filter(None, [doc.get("username", ""), doc.get("bio", ""), doc.get("category", "")] ))
    try:
        emb = get_embedding(text)
    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    influencers_collection.update_one({"_id": doc["_id"]}, {"$set": {"embedding": emb, "embedding_updated_at": datetime.utcnow()}})
    return {"updated": True, "_id": str(doc.get("_id"))}


@router.post("/create")
def create_influencer(payload: Dict[str, Any], current_user: dict = Depends(get_current_user)):
    """Create a local influencer document (useful for testing).

    Example body:
    {
      "username": "the_foodigram",
      "bio": "Hyderabad-based food blogger...",
      "followers": 45000,
      "category": "food"
    }
    """
    if influencers_collection is None:
        raise HTTPException(status_code=500, detail="MongoDB influencers collection not configured")
    try:
        res = influencers_collection.insert_one(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"inserted_id": str(res.inserted_id)}



@router.get("/search/semantic")
def semantic_search(query: str, top_k: int = 10, expand: bool = True):
    if influencers_collection is None:
        raise HTTPException(status_code=500, detail="MongoDB influencers collection not configured")
    if not query:
        raise HTTPException(status_code=400, detail="query is required")

    # 1) expand terms (cap to 15)
    expanded_terms = []
    if expand and OPENAI_KEY:
        try:
            expanded_terms = _expand_query_with_gpt(query)[:15]
        except Exception:
            expanded_terms = []

    # 2) get / cache embedding for the query
    try:
        q_emb = get_embedding(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3) build a text query to prune candidates (top N)
    text_terms = [query] + expanded_terms
    # sanitize and dedupe search terms to avoid malformed or huge regexes
    text_terms = _sanitize_search_terms(text_terms, max_terms=12)

    mongo_query = {"embedding": {"$exists": True}}
    if text_terms:
        # If a text index exists, prefer a simple $text query (avoid mixing many regexes)
        has_text_index = HAS_TEXT_INDEX
        if has_text_index:
            try:
                mongo_query = {"$text": {"$search": " ".join(text_terms)}, "embedding": {"$exists": True}}
            except Exception as e:
                print("$text clause build failed, falling back to regexes:", e)
                has_text_index = False

        if not has_text_index:
            # build separate regex clauses per term (word-boundary) rather than a single giant regex
            or_clauses = []
            for term in text_terms:
                if not term:
                    continue
                try:
                    pattern = r"\\b" + re.escape(term) + r"\\b"
                except Exception:
                    pattern = re.escape(term)
                or_clauses.append({"username": {"$regex": pattern, "$options": "i"}})
                or_clauses.append({"bio": {"$regex": pattern, "$options": "i"}})
                or_clauses.append({"category": {"$regex": pattern, "$options": "i"}})

            if or_clauses:
                mongo_query["$or"] = or_clauses

    # limit candidate pool for speed
    CANDIDATE_LIMIT = max(200, top_k * 20)
    # debug log the query being executed to help troubleshoot planner issues
    try:
        print("[DEBUG] executing mongo query:", mongo_query)
        cursor = influencers_collection.find(
            mongo_query,
            {"username": 1, "bio": 1, "followers": 1, "embedding": 1, "category": 1}
        ).limit(CANDIDATE_LIMIT)
        candidates = list(cursor)
    except Exception as e:
        # if the query planner fails (NoQueryExecutionPlans / OperationFailure), fall back to a simpler query
        print("planner returned error :: caused by ::", e)
        try:
            # try a simple username-only regex using the first term, if present
            simple_query = {"embedding": {"$exists": True}}
            if text_terms:
                first = text_terms[0]
                if first:
                    simple_query["username"] = {"$regex": re.escape(first), "$options": "i"}
            print("[DEBUG] retrying with simpler query:", simple_query)
            cursor = influencers_collection.find(simple_query, {"username": 1, "bio": 1, "followers": 1, "embedding": 1, "category": 1}).limit(CANDIDATE_LIMIT)
            candidates = list(cursor)
        except Exception as e2:
            print("failed fallback query as well:", e2)
            raise HTTPException(status_code=500, detail=str(e))

    # 4) score each candidate
    scored = []
    # precompute for keyword normalization
    lower_terms = [t.lower() for t in expanded_terms] if expanded_terms else []
    max_kw = 1
    kw_counts = []

    for d in candidates:
        cos = _cosine_similarity(q_emb, d.get("embedding", []))
        kw = 0
        if lower_terms:
            text = " ".join(filter(None, [d.get("username",""), d.get("bio",""), d.get("category","")])).lower()
            kw = sum(1 for t in lower_terms if t in text)
        kw_counts.append(kw)
        scored.append({"doc": d, "cos": float(cos), "kw": kw})

    if scored:
        max_kw = max(kw_counts) or 1

    for s in scored:
        kw_norm = s["kw"] / max_kw
        s["score"] = 0.8 * s["cos"] + 0.2 * kw_norm

    scored.sort(key=lambda x: x["score"], reverse=True)

    results = [
        {
            "score": round(s["score"], 6),
            "cosine": round(s["cos"], 6),
            "kw_matches": s["kw"],
            "username": s["doc"].get("username"),
            "bio": s["doc"].get("bio"),
            "followers": s["doc"].get("followers"),
            "category": s["doc"].get("category"),
        }
        for s in scored[:int(top_k)]
    ]

    return {"results": results, "query": query, "expanded_terms": expanded_terms}



def _expand_query_with_gpt(query: str) -> list[str]:
    """Use OpenAI chat completion to expand the user's query into related keywords/phrases.

    Returns a list of short tokens (keywords). This function is intentionally simple and expects
    a JSON-array or newline-separated plain text response; it will try parsing both.
    """
    if not OPENAI_KEY:
        return []

    system = "You are a helpful assistant that expands short influencer search queries into a list of related keywords, locations, synonyms, and hashtags. Return a JSON array of short keywords."
    user = f"Expand the following query into related keywords, synonyms, locations and hashtags: \"{query}\". Return only a JSON array of strings like [\"food\", \"Hyderabad\", \"foodie\"]"
    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=20)
    if resp.status_code != 200:
        # fail silently and return no terms
        return []
    text = ""
    try:
        data = resp.json()
        text = data.get("choices", [])[0].get("message", {}).get("content", "")
    except Exception:
        text = resp.text

    # normalize common wrapper formats (remove markdown code fences, surrounding text)
    try:
        # remove leading/trailing ``` blocks and language tags like ```json
        text = re.sub(r"^```[\w-]*\n", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\n```$", "", text.strip(), flags=re.IGNORECASE)
        # If the model returned a JSON array inside text (possibly with extra text), try to extract it
        first_bracket = text.find("[")
        last_bracket = text.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            candidate = text[first_bracket:last_bracket+1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
            except Exception:
                # fallthrough to try parsing the whole text
                pass

    except Exception:
        # Ignore normalization errors and continue to try parsing raw text
        pass

    # try JSON parse of the raw text as a final fallback
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if isinstance(x, str) and x.strip()]
    except Exception:
        pass

    # fallback: split by newlines/commas and return short tokens
    tokens = re.split(r"[\n,;]+", text)
    results = []
    for t in tokens:
        t = t.strip().strip('"').strip()
        if not t:
            continue
        # keep short tokens only
        if len(t) > 1 and len(t) < 50:
            # if the token has a dash like - Hyd -> Hyd
            parts = re.split(r"[:\-\(\)]+", t)
            results.append(parts[0].strip())
    # dedupe while preserving order
    deduped = []
    seen = set()
    for r in results:
        low = r.lower()
        if low in seen:
            continue
        seen.add(low)
        deduped.append(r)
    return deduped


def _sanitize_search_terms(terms: list[str], max_terms: int = 12) -> list[str]:
    """Clean and limit expanded search terms to avoid producing huge/invalid regexes.

    - Removes backticks, brackets and other punctuation
    - Keeps only alphanumeric, spaces and hyphens
    - Trims to at most `max_terms` and dedupes while preserving order
    - Limits term length to 40 chars and word count to 4
    """
    cleaned: list[str] = []
    seen = set()
    for t in terms:
        if not t:
            continue
        s = t.strip()
        # remove explicit backticks/brackets/quotes and control chars
        s = re.sub(r"[`\"\[\]\{\}\(\)<>]", " ", s)
        # replace any non-word/space/hyphen with space
        s = re.sub(r"[^\w\s\-]", " ", s)
        # collapse whitespace
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue
        # limit length and words
        if len(s) > 40:
            # keep first 40 chars worth of words
            parts = s.split()
            s = " ".join(parts[:4])
        words = s.split()
        if len(words) > 4:
            s = " ".join(words[:4])
        s_low = s.lower()
        if s_low in seen:
            continue
        seen.add(s_low)
        cleaned.append(s_low)
        if len(cleaned) >= max_terms:
            break
    return cleaned



@router.post("/enrich/batch_embeddings")
def enrich_batch(limit: int = 500, current_user: dict = Depends(get_current_user)):
    if influencers_collection is None:
        raise HTTPException(status_code=500, detail="MongoDB influencers collection not configured")

    docs = list(influencers_collection.find(
        {"$or": [{"embedding": {"$exists": False}}, {"embedding": []}]},
        {"_id": 1, "username": 1, "bio": 1, "category": 1}
    ).limit(int(limit)))

    updated = 0
    for d in docs:
        text = " ".join(filter(None, [d.get("username",""), d.get("bio",""), d.get("category","")]))
        if not text.strip():
            continue
        try:
            emb = get_embedding(text)
        except Exception:
            continue
        influencers_collection.update_one({"_id": d["_id"]}, {"$set": {"embedding": emb, "embedding_updated_at": datetime.utcnow()}})
        updated += 1

    return {"updated": updated, "attempted": len(docs)}






@router.get("/search/top")
def search_top_influencers(keyword: str, limit: int = 10, user_id: str | None = None):
    """
    Search top influencers by keyword using Mongo cache + RapidAPI.
    - Cache key: normalized keyword + limit
    - First tries exact normalized lookup (keyword lowercased + int limit).
    - If not found, tries case-insensitive lookup with same limit.
    - Stores normalized keyword + raw keyword + limit when saving.
    """
    if not keyword:
        raise HTTPException(status_code=400, detail="keyword is required")

    raw_keyword = keyword.strip()
    key = raw_keyword.lower()
    # include user_id in cache key when provided so cached results are user-scoped
    cache_query = {"keyword": key, "limit": int(limit)}
    if user_id:
        cache_query["user_id"] = str(user_id)

    # Try exact cached entry first -> return immediate if found
    if searches_collection is not None:
        try:
            cached = searches_collection.find_one(cache_query)
            if cached and "results" in cached:
                return {"results": cached["results"], "cached": True}
            # fallback: case-insensitive keyword match (same limit) — include user_id if present
            regex_q = {"keyword": {"$regex": f"^{re.escape(raw_keyword)}$", "$options": "i"}, "limit": int(limit)}
            if user_id:
                regex_q["user_id"] = str(user_id)
            cached = searches_collection.find_one(regex_q)
            if cached and "results" in cached:
                return {"results": cached["results"], "cached": True}
        except Exception as e:
            print("search cache lookup error:", e)

    # Fetch from RapidAPI when not cached
    url = "https://instagram-best-experience.p.rapidapi.com/users_search"
    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    params = {"query": raw_keyword, "count": limit}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15.0)
    except Exception as e:
        # If API fails, try to return any cached entry (ignore limit) before failing
        if searches_collection is not None:
            try:
                fallback = searches_collection.find_one({"keyword": {"$regex": f"^{re.escape(raw_keyword)}$", "$options": "i"}})
                if fallback and "results" in fallback:
                    return {"results": fallback["results"], "cached": True, "stale": True}
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=f"RapidAPI request error: {e}")

    if resp.status_code != 200:
        # try fallback cache before raising
        if searches_collection is not None:
            try:
                fallback = searches_collection.find_one({"keyword": {"$regex": f"^{re.escape(raw_keyword)}$", "$options": "i"}})
                if fallback and "results" in fallback:
                    return {"results": fallback["results"], "cached": True, "stale": True}
            except Exception:
                pass
        raise HTTPException(status_code=502, detail=f"RapidAPI error: {resp.text}")

    try:
        data = resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from RapidAPI: {e}")

    if isinstance(data, dict) and "users" in data:
        users_list = data["users"]
    elif isinstance(data, list):
        users_list = data
    else:
        users_list = []

    results = []
    for user in users_list[:limit]:
        if not isinstance(user, dict):
            continue

        pk = user.get("pk") or user.get("id")
        username = user.get("username")
        profile = {
            "pk": pk,
            "username": username,
            "full_name": user.get("full_name") or user.get("name"),
            "followers": user.get("follower_count"),
            "profile_pic": user.get("profile_pic_url"),
            "bio": user.get("biography") or user.get("bio", ""),
        }

        # Enrich per-user: try to fetch profile + insights (best-effort)
        try:
            prof = None
            try:
                prof = fetch_rapid_follower_profile(pk) if pk else None
            except Exception:
                prof = None

            insights = None
            try:
                insights = get_insights(user_id=pk) if pk else None
            except Exception:
                insights = None

            if insights:
                profile.update({
                    "post_count": insights.get("post_count"),
                    "avg_likes": insights.get("avg_likes"),
                    "engagement": insights.get("engagement"),
                    "engagement_rate_percent": insights.get("engagement_rate_percent"),
                    "followers": insights.get("followers") or profile.get("followers"),
                    "total_posts": insights.get("total_posts") or (prof.get("media_count") if prof else None),
                })
            else:
                if prof:
                    profile.update({
                        "followers": prof.get("follower_count") or profile.get("followers"),
                        "total_posts": prof.get("media_count"),
                        "post_count": None,
                        "avg_likes": None,
                        "engagement": None,
                        "engagement_rate_percent": None,
                    })
        except Exception as e:
            print(f"enrichment error for {username or pk}: {e}")

        results.append(profile)
        time.sleep(0.25)

    # Merge semantic search results into RapidAPI results (best-effort)
    expanded_terms: list[str] = []
    try:
        merged_results, expanded_terms = _merge_results_with_semantic(results, raw_keyword, limit)
        results = merged_results
    except Exception as e:
        print("semantic merge error:", e)

    # Save to Mongo (best-effort) — store normalized keyword + raw + limit
    if searches_collection is not None:
        try:
            doc = {
                "keyword": key,
                "keyword_raw": raw_keyword,
                "limit": int(limit),
                "results": results,
                "expanded_terms": expanded_terms,
                "created_at": datetime.utcnow(),
            }
            # attach user_id to stored doc when provided
            if user_id:
                doc["user_id"] = str(user_id)
            # use normalized cache_query to upsert so subsequent exact lookups succeed
            searches_collection.replace_one(cache_query, doc, upsert=True)
        except Exception as e:
            print("search cache write error:", e)

    return {"results": results, "cached": False, "expanded_terms": expanded_terms}



@router.get("/insights")
def user_insights(username: str | None = None, media_id: str | None = None, user_id: str | None = None, current_user: dict = Depends(get_current_user)):
    """
    Client endpoint. Accepts:
      - ?user_id=... -> fetch aggregated feed metrics for that user (preferred)
      - ?username=... -> will try to resolve user_id from profile (may be slower)
      - media_id is ignored in this aggregated endpoint (feed aggregation)
    Examples:
      GET /influencers/insights?user_id=13460080
      GET /influencers/insights?username=_the_foodigram001
    """
    print(f"[DEBUG] /influencers/insights called with user_id={user_id}, username={username}, media_id={media_id}")
    try:
        metrics = get_insights(username=username, media_id=media_id, user_id=user_id)
        print(f"[DEBUG] /influencers/insights result for user_id={user_id}, username={username}: {metrics}")
    except HTTPException as e:
        print(f"[DEBUG] /influencers/insights HTTPException for user_id={user_id}, username={username}: {e.detail}")
        raise
    except Exception as e:
        print(f"[DEBUG] /influencers/insights error for user_id={user_id}, username={username}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return metrics

def get_insights(username: str = None, media_id: str | None = None, user_id: str | None = None) -> dict:
    """
    Fetch aggregated feed insights for a user (uses user_id / pk).
    Only fetches last 20 posts to avoid rate limits.
    Returns: avg_likes, engagement, engagement_rate_percent, post_count.
    """
    print(f"[DEBUG] get_insights called with user_id={user_id}, username={username}, media_id={media_id}")
    # resolve user_id if only username is provided

    

    if not user_id:
        
        print(f"[DEBUG] get_insights missing user_id for username={username}")
        raise HTTPException(status_code=400, detail="user_id (pk) is required to fetch feed insights.")

    headers = {
        "x-rapidapi-host": RAPIDAPI_HOST,
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    feed_url = f"https://{RAPIDAPI_HOST}/feed"
    params = {"user_id": str(user_id), "count": 20}   # ✅ only last 20 posts

    def fetch_and_parse():
        try:
            print(f"[DEBUG] get_insights requesting feed: {feed_url} params={params}")
            resp = requests.get(feed_url, headers=headers, params=params, timeout=20.0)
            time.sleep(0.5)  # Add delay after feed request
        except Exception as e:
            print(f"[DEBUG] get_insights RapidAPI request error (feed): {e}")
            raise HTTPException(status_code=502, detail=f"RapidAPI request error (feed): {e}")

        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            print(f"[DEBUG] get_insights RapidAPI error (feed): {err}")
            raise HTTPException(status_code=502, detail=f"RapidAPI error (feed): {err}")

        data = resp.json()
        print(f"[DEBUG] get_insights feed data received: {str(data)[:300]}...")  # Print first 300 chars

        items = []
        if isinstance(data, dict):
            if "items" in data:
                items = data["items"]
            elif "media" in data:
                items = data["media"]
            elif "data" in data:
                items = data["data"]

        total_likes = 0
        total_comments = 0
        post_count = 0

        for it in items:
            if not isinstance(it, dict):
                continue
            likes = int(it.get("like_count", 0))
            comments = int(it.get("comment_count", 0))
            total_likes += likes
            total_comments += comments
            post_count += 1

        avg_likes = int(total_likes / post_count) if post_count else 0
        engagement = total_likes + total_comments

        # ---- Fetch followers count ----
        followers = None
        media_count = None
        try:
            profile_data = fetch_rapid_follower_profile(user_id)
            followers = profile_data.get("follower_count")
            media_count = profile_data.get("media_count")
            print(f"[DEBUG] get_insights fetched profile_data for user_id={user_id}: {profile_data}")
        except Exception as e:
            print(f"[DEBUG] get_insights failed to fetch profile_data for user_id={user_id}: {e}")
            followers = None
            media_count = None

        # engagement rate
        engagement_rate = None
        if followers and followers > 0:
            engagement_rate = round((engagement / followers) * 100, 2)

        result = {
            "post_count": post_count,
            "avg_likes": avg_likes,
            "engagement": engagement,
            "engagement_rate_percent": engagement_rate,
            "followers": followers,
            "total_posts": media_count,
            
        }
        print(f"[DEBUG] get_insights result for user_id={user_id}: {result}")
        return result

    # First attempt
    result = fetch_and_parse()

    # Fallback: If followers or engagement_rate_percent is None, retry once after 2s delay
    if result.get("followers") is None or result.get("engagement_rate_percent") is None:
        print(f"[DEBUG] get_insights missing followers or engagement_rate_percent, retrying after 2s...")
        time.sleep(2)
        result = fetch_and_parse()

    return result


@router.get("/profile")
def fetch_rapid_follower_profile(user_id: str, current_user: dict = Depends(get_current_user)) -> dict:
    """
    Fetch profile info from RapidAPI /profile endpoint.
    Returns {follower_count, media_count, username, full_name, ...}
    """
    print(f"[DEBUG] fetch_rapid_follower_profile called with user_id={user_id}")
    if not RAPIDAPI_KEY:
        print(f"[DEBUG] fetch_rapid_follower_profile missing RAPIDAPI_KEY")
        raise HTTPException(status_code=500, detail="No RAPIDAPI_KEY configured")

    url = "https://instagram-best-experience.p.rapidapi.com/profile"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }
    params = {"user_id": str(user_id)}

    try:
        print(f"[DEBUG] fetch_rapid_follower_profile requesting: {url} params={params}")
        resp = requests.get(url, headers=headers, params=params, timeout=20.0)
        time.sleep(2)  # <-- Add a 2 second delay after the API call
    except Exception as e:
        print(f"[DEBUG] fetch_rapid_follower_profile RapidAPI request error: {e}")
        raise HTTPException(status_code=502, detail=f"RapidAPI request error (profile): {e}")

    if resp.status_code != 200:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        print(f"[DEBUG] fetch_rapid_follower_profile RapidAPI error: {err}")
        raise HTTPException(status_code=502, detail=f"RapidAPI error (profile): {err}")

    try:
        data = resp.json()
        print(f"[DEBUG] fetch_rapid_follower_profile data received: {data}")
    except Exception as e:
        print(f"[DEBUG] fetch_rapid_follower_profile Invalid JSON: {e}")
        raise HTTPException(status_code=502, detail=f"Invalid JSON from RapidAPI (profile): {e}")

    return {
        "user_id": data.get("pk"),
        "username": data.get("username"),
        "full_name": data.get("full_name"),
        "follower_count": data.get("follower_count"),
        "media_count": data.get("media_count"),
        "profile_pic_url": data.get("profile_pic_url"),
        "bio": data.get("biography"),
    }
    
    


@router.get("/fetch_rapid_followers")
def get_rapid_followers(user_id: str, next_max_id: str | None = None):
    """
    GET /influencers/fetch_rapid_followers?user_id=12345[&next_max_id=...]
    Returns RapidAPI followers data for the given user_id.
    This endpoint is specifically designed for frontend integration.
    """
    try:
        followers_data = fetch_rapid_follower_profile(user_id=user_id)
        return followers_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SummaryRequest(BaseModel):
    username: str
    bio: str | None = None
    # metrics (optional but recommended for deep analysis)
    post_count: int | None = None
    avg_likes: int | None = None
    engagement: int | None = None
    engagement_rate_percent: float | None = None
    followers: int | None = None
    total_posts: int | None = None
    user_id: str | None = None
    full_name: str | None = None
    follower_count: int | None = None
    media_count: int | None = None
    profile_pic_url: str | None = None

@router.post("/summary")
def generate_summary(request: SummaryRequest):
    """
    Generates an in-depth (2-3 page) human-friendly analysis of an influencer.
    Uses provided metrics (if available) to analyze engagement, reach and recommend
    campaign ideas, pricing guidance and next steps.
    """
    if not OPENAI_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    # System prompt instructs style, structure and desired length (2-3 pages)
    system_prompt = (
        "You are a senior influencer marketing analyst and copywriter. Produce a detailed, "
        "human-friendly report approximately 2-3 paragraphs long (aim for ~400-600 words) based on the "
        "profile and metrics provided. Use simple, conversational language (not robotic). Organize the "
        "output with clear headings and subsections. Sections must include: Executive Summary, Audience & Reach, "
        "Engagement Analysis (use provided metrics and explain what they mean), Content Strategy & Strengths, "
        "Weaknesses & Risks, Recommended Collaboration Types & Creative Angles, Pricing Guidance (estimate), "
        "Actionable Next Steps, and an Appendix with raw metrics. When metrics are missing, explicitly note that and "
        "qualify recommendations. End with 5 concise bullet-point next steps for a brand. Be practical and tactical."
        "Pricing Fairness Check: Compare the calculated price-per-post ($<price_per_post>) to the influencer’s typical quote (if provided) and state whether it appears fair, 10–20 % above market, or a bargain."
        "Campaign Forecast: Estimate likely reach (followers × engagement rate × 0.8) and ballpark clicks (reach × 2 %) for a single sponsored post."
    )

    # Build a detailed user prompt including all available metrics
    parts = [f"Username: @{request.username}"]
    if request.full_name:
        parts.append(f"Full name: {request.full_name}")
    if request.user_id:
        parts.append(f"User ID: {request.user_id}")
    if request.follower_count is not None:
        parts.append(f"Follower count (profile): {request.follower_count}")
    if request.followers is not None:
        parts.append(f"Followers (enriched): {request.followers}")
    if request.media_count is not None:
        parts.append(f"Total posts (profile): {request.media_count}")
    if request.total_posts is not None:
        parts.append(f"Total posts (enriched): {request.total_posts}")
    if request.post_count is not None:
        parts.append(f"Recent posts considered: {request.post_count}")
    if request.avg_likes is not None:
        parts.append(f"Average likes per post: {request.avg_likes}")
    if request.engagement is not None:
        parts.append(f"Total engagement (likes+comments over sample): {request.engagement}")
    if request.engagement_rate_percent is not None:
        parts.append(f"Engagement rate (%): {request.engagement_rate_percent}")
    if request.profile_pic_url:
        parts.append(f"Profile image URL: {request.profile_pic_url}")
    parts.append(f"Bio: {request.bio or 'N/A'}")

    user_prompt = (
        "Please analyze the influencer using the data below and produce the requested 2-3 page report.\n\n"
        + "\n".join(parts)
        + "\n\nDeliverable notes: Use the metrics to calculate and interpret engagement quality, "
        "audience relevance, and likely content performance. Provide realistic collaboration ideas and "
        "give a pricing range (low/typical/high) with rationale. Keep tone friendly, actionable and easy to read."
    )

    body = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2000,  # allows for long output (adjust if using a different model/token limits)
    }

    try:
        headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=60.0)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request error: {e}")

    if resp.status_code != 200:
        # return helpful debugging info while avoiding leaking keys
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {resp.text}")

    data = resp.json()
    summary = ""
    try:
        summary = data.get("choices", [])[0].get("message", {}).get("content", "").strip()
    except Exception:
        summary = ""

    if not summary:
        raise HTTPException(status_code=502, detail="Failed to generate summary")

    return {"username": request.username, "summary": summary}
