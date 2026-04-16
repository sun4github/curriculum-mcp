#!/usr/bin/env python3
"""
Curriculum Ingestion Script
===========================
Reads photos of textbook/workbook pages from a folder, extracts structured Q&A
using a multimodal LLM (vision), and inserts them into a PostgreSQL database.

Folder layout:
  book-folder/
    page1.jpg              <- loose image: AI auto-detects the topic
    page2.jpg
    1/                     <- subfolder: AI auto-detects topic from page images
      page1.jpg
      page2.jpg
    2/
      page1.jpg
    3/                    <- just a numeric name; topic comes from AI
      page1.jpg

  Most exercises are a single page — just drop them loose in the folder.
  For multi-page exercises, group them in a numeric subfolder (1, 2, 3…).
  The AI will detect the topic from the page images, same as loose images.

Usage:
  python3 ingest.py --folder ./math-workbook --subject math [--dry-run]
  python3 ingest.py --folder ./math-workbook --subject math --immediate   # skip batch, call LLM one-by-one
  python3 ingest.py --folder ./math-workbook --subject math --resume-batch <batch_id>  # resume a batch

Config:
  Copy config.example.yaml to config.yaml and customize prompts + DB settings.
  Set INGEST_LLM_API_KEY env var (or put it in config.yaml).

Batch API mode (default):
  All images are submitted as a single OpenAI Batch API job (50 % discount, up to
  24 h completion window).  The script polls every ~30 s and blocks until the
  batch finishes, then processes results and writes to DB.  Use --immediate to
  fall back to the original sequential call behaviour.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import psycopg2
import psycopg2.extras
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config/config.yaml"
EXAMPLE_CONFIG_PATH = Path(__file__).parent / "config.example.yaml"


def load_config():
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found. Copy {EXAMPLE_CONFIG_PATH.name} to config.yaml and customize.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# LLM Call
# ---------------------------------------------------------------------------

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_map.get(ext, "image/jpeg")


def build_request_body(config: dict, image_path: str, subject: str, topic: str | None) -> dict:
    """Build the chat/completions request body (shared by immediate and batch modes)."""
    llm_cfg = config["llm"]
    b64 = encode_image(image_path)
    mime = get_image_mime(image_path)

    system_prompt = config["prompts"]["system"]
    user_prompt = config["prompts"]["user"]
    user_prompt = user_prompt.replace("{{SUBJECT}}", subject)
    user_prompt = user_prompt.replace("{{TOPIC}}", topic or "auto-detect")
    user_prompt = user_prompt.replace("{{FILENAME}}", Path(image_path).name)

    return {
        "model": llm_cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ],
        "temperature": llm_cfg.get("temperature", 0.1),
        "max_tokens": llm_cfg.get("max_tokens", 4096),
        "response_format": {"type": "json_object"},
    }


def call_llm(config: dict, image_path: str, subject: str, topic: str | None) -> dict:
    """Send image to multimodal LLM immediately and return structured Q&A dict."""
    llm_cfg = config["llm"]
    api_key = os.getenv("INGEST_LLM_API_KEY") or llm_cfg.get("api_key", "")
    base_url = llm_cfg["base_url"].rstrip("/")

    payload = build_request_body(config, image_path, subject, topic)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    url = f"{base_url}/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"]
    print(f"  [LLM RAW RESPONSE]\n{content}\n")
    return json.loads(content)


def get_batch_base_url(config: dict) -> str:
    """Resolve Batch API base URL from config (llm.batch_url)."""
    llm_cfg = config["llm"]
    batch_url = llm_cfg.get("batch_url")
    if not batch_url:
        raise ValueError("Missing llm.batch_url in config.yaml")
    return batch_url.rstrip("/")


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def build_batch_jsonl(config: dict, registry: dict, subject: str) -> str:
    """Return JSONL string — one request per line — for the OpenAI Batch API."""
    lines = []
    for custom_id, meta in registry.items():
        body = build_request_body(config, str(meta["image_path"]), subject, meta["topic"])
        lines.append(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))
    return "\n".join(lines)


def upload_batch_file(batch_base_url: str, api_key: str, jsonl_content: str) -> str:
    """Upload JSONL to OpenAI Files API with purpose=batch. Returns file_id."""
    url = f"{batch_base_url}/files"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.post(
        url,
        headers=headers,
        files={
            "file": ("batch_requests.jsonl", io.BytesIO(jsonl_content.encode()), "application/jsonl"),
            "purpose": (None, "batch"),
        },
        timeout=120,
    )
    resp.raise_for_status()
    file_id = resp.json()["id"]
    print(f"  Uploaded batch input file: {file_id}")
    return file_id


def create_batch(batch_base_url: str, api_key: str, file_id: str) -> str:
    """Submit a batch job. Returns batch_id."""
    url = f"{batch_base_url}/batches"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        url,
        headers=headers,
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
        timeout=30,
    )
    resp.raise_for_status()
    batch_id = resp.json()["id"]
    print(f"  Created batch job: {batch_id}")
    return batch_id


def poll_batch(batch_base_url: str, api_key: str, batch_id: str, poll_interval: int = 30) -> str:
    """Block until batch completes. Returns output_file_id."""
    url = f"{batch_base_url}/batches/{batch_id}"
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"Polling batch {batch_id} every {poll_interval}s …")
    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]
        counts = data.get("request_counts", {})
        completed = counts.get("completed", 0)
        failed = counts.get("failed", 0)
        total = counts.get("total", 0)
        print(f"  [{time.strftime('%H:%M:%S')}] status={status}  {completed}/{total} done, {failed} failed")

        if status == "completed":
            return data["output_file_id"]
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {status}")

        time.sleep(poll_interval)


def download_batch_results(batch_base_url: str, api_key: str, output_file_id: str) -> dict:
    """Download batch output JSONL. Returns dict[custom_id -> parsed result | error dict]."""
    url = f"{batch_base_url}/files/{output_file_id}/content"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()

    results = {}
    for line in resp.text.strip().splitlines():
        if not line:
            continue
        obj = json.loads(line)
        custom_id = obj["custom_id"]
        if obj.get("error"):
            results[custom_id] = {"_error": obj["error"]}
        else:
            try:
                content = obj["response"]["body"]["choices"][0]["message"]["content"]
                results[custom_id] = json.loads(content)
            except (KeyError, json.JSONDecodeError) as exc:
                results[custom_id] = {"_error": str(exc)}
    print(f"  Downloaded {len(results)} results from output file {output_file_id}")
    return results


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db_connection(config: dict):
    db_cfg = config["database"]
    return psycopg2.connect(
        host=db_cfg.get("host", "localhost"),
        port=db_cfg.get("port", 5432),
        dbname=db_cfg["dbname"],
        user=db_cfg["user"],
        password=db_cfg.get("password", "") or os.getenv("INGEST_DB_PASSWORD", ""),
    )


def ensure_schema(conn):
    """Create tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS subjects (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id SERIAL PRIMARY KEY,
                subject_id INTEGER REFERENCES subjects(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                UNIQUE(subject_id, name)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id SERIAL PRIMARY KEY,
                topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                context TEXT,
                instructions TEXT,
                source_page TEXT,
                sort_order INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id SERIAL PRIMARY KEY,
                exercise_id INTEGER REFERENCES exercises(id) ON DELETE CASCADE,
                difficulty INTEGER DEFAULT 3 CHECK (difficulty BETWEEN 1 AND 5),
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                explanation TEXT,
                choices JSONB,
                sort_order INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id SERIAL PRIMARY KEY,
                student_id INTEGER REFERENCES students(id) ON DELETE CASCADE,
                question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE,
                student_name TEXT NOT NULL DEFAULT 'rikki',
                correct BOOLEAN NOT NULL,
                time_seconds INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_exercises_topic ON exercises(topic_id);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_questions_exercise ON questions(exercise_id);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_attempts_question ON attempts(question_id);
        """)
    conn.commit()


def upsert_subject(conn, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO subjects (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            (name,),
        )
        return cur.fetchone()[0]


def upsert_topic(conn, subject_id: int, name: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO topics (subject_id, name) VALUES (%s, %s) ON CONFLICT (subject_id, name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            (subject_id, name),
        )
        return cur.fetchone()[0]


def insert_exercise(conn, topic_id: int, title: str, context: str | None, instructions: str, source_page: str, sort_order: int) -> int:
    """Insert an exercise and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO exercises (topic_id, title, context, instructions, source_page, sort_order)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (topic_id, title, context, instructions, source_page, sort_order),
        )
        conn.commit()
        return cur.fetchone()[0]


def insert_questions(conn, exercise_id: int, questions: list[dict], source_page: str) -> int:
    """Insert questions, return count inserted."""
    inserted = 0
    with conn.cursor() as cur:
        for i, q in enumerate(questions):
            difficulty = q.get("difficulty", 3)
            question_text = q.get("question", "")
            answer = q.get("answer", "")
            explanation = q.get("explanation", "")
            choices = json.dumps(q.get("choices", [])) if q.get("choices") else None

            try:
                cur.execute(
                    """INSERT INTO questions (exercise_id, difficulty, question, answer, explanation, choices, sort_order)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (exercise_id, difficulty, question_text, answer, explanation, choices, i + 1),
                )
                if cur.rowcount > 0:
                    inserted += 1
            except psycopg2.Error as e:
                print(f"  WARN: Skipping question: {e}")
                conn.rollback()
                continue
    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Main Ingestion
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def find_images(folder: Path) -> list[Path]:
    images = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return images


def discover_topics(folder: Path) -> list[tuple[str | None, list[Path], bool]]:
    """Walk folder and return (topic_name, [image_paths], should_merge) tuples.

    Subfolders: topic=None (AI auto-detects from page images), should_merge=True.
    Loose images in root: topic=None (AI auto-detects per-page), should_merge=False.

    Folder naming convention:
      1/   <- subfolder: just a numeric name, topic detected by AI from images
      2/
      3/
    """
    topics: list[tuple[str | None, list[Path], bool]] = []

    # Loose images in root -> topic=None, don't merge
    root_images = find_images(folder)
    if root_images:
        topics.append((None, root_images, False))

    # Each subfolder -> topic=None (AI detects from images), merge pages
    for child in sorted(folder.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            imgs = find_images(child)
            if imgs:
                topics.append((None, imgs, True))

    return topics


def build_image_registry(topics: list) -> dict:
    """Return OrderedDict[custom_id -> metadata] for all images across all topic groups."""
    registry = OrderedDict()
    idx = 0
    for topic_name, images, should_merge in topics:
        group_key = str(images[0].parent) if should_merge else None
        for page_num, img_path in enumerate(images):
            custom_id = f"req-{idx:04d}"
            registry[custom_id] = {
                "image_path": img_path,
                "topic": topic_name,          # None → AI auto-detect
                "group_key": group_key or str(img_path),
                "should_merge": should_merge,
                "page_num": page_num,
            }
            idx += 1
    return registry


def _merge_group(pages: list[tuple[str, dict]], llm_results: dict) -> dict | None:
    """Merge LLM results for a multi-page subfolder into one exercise dict."""
    merged_context = None
    merged_instructions = ""
    merged_title = None
    merged_topic = None
    merged_questions: list[dict] = []
    errors = 0

    for custom_id, meta in pages:
        result = llm_results.get(custom_id)
        if result is None or "_error" in result:
            print(f"  WARN: missing/error result for {meta['image_path'].name}: {result}")
            errors += 1
            continue

        if not merged_topic:
            merged_topic = result.get("topic")

        exercises = result.get("exercises", [])
        questions = result.get("questions", [])
        if not exercises and questions:
            fallback_title = merged_topic or "general"
            exercises = [{"title": fallback_title, "instructions": "", "questions": questions}]

        for ex in exercises:
            ex_questions = ex.get("questions", [])
            if not ex_questions:
                continue
            if merged_context is None and ex.get("context"):
                merged_context = ex["context"]
            if not merged_instructions and ex.get("instructions"):
                merged_instructions = ex["instructions"]
            if not merged_title and ex.get("title"):
                merged_title = ex["title"]
            merged_questions.extend(ex_questions)

        total_q = sum(len(ex.get("questions", [])) for ex in exercises)
        print(f"  Page {meta['image_path'].name}: {len(exercises)} exercise(s), {total_q} questions | topic={merged_topic or 'auto-detect'}")

    if not merged_topic:
        merged_topic = "general"
    if not merged_title:
        merged_title = merged_topic

    if not merged_questions:
        return None

    return {
        "topic": merged_topic,
        "title": merged_title,
        "context": merged_context,
        "instructions": merged_instructions,
        "questions": merged_questions,
    }


def _process_all_results(
    llm_results: dict,
    registry: dict,
    conn,
    subject_id: int | None,
    dry_run: bool,
) -> tuple[int, int]:
    """Group results by group_key, merge subfolders, and insert into DB (or print for dry-run)."""
    # Group by group_key preserving insertion order
    groups: OrderedDict[str, dict] = OrderedDict()
    for custom_id, meta in registry.items():
        gk = meta["group_key"]
        if gk not in groups:
            groups[gk] = {"should_merge": meta["should_merge"], "pages": []}
        groups[gk]["pages"].append((custom_id, meta))

    total_inserted = 0
    total_errors = 0

    for group_key, group in groups.items():
        should_merge = group["should_merge"]
        pages = group["pages"]

        if should_merge:
            # ----- Subfolder: merge all pages into one exercise -----
            folder_label = Path(group_key).name
            print(f"--- Subfolder '{folder_label}' ({len(pages)} pages) ---")
            merged = _merge_group(pages, llm_results)
            if merged is None:
                print(f"  No questions extracted.")
                total_errors += 1
                continue

            if dry_run:
                print(f"  Merged exercise: {merged['title']} (Topic: {merged['topic']})")
                if merged.get("context"):
                    print(f"    Context: {merged['context'][:120]}…")
                if merged.get("instructions"):
                    print(f"    Instructions: {merged['instructions'][:80]}")
                for j, q in enumerate(merged["questions"], 1):
                    print(f"    Q{j}: {q.get('question', '')[:80]}…")
                    print(f"    A{j}: {q.get('answer', '')[:80]}")
                continue

            topic_id = upsert_topic(conn, subject_id, merged["topic"])
            source_pages = "+".join(meta["image_path"].stem for _, meta in pages)
            try:
                exercise_id = insert_exercise(
                    conn, topic_id, merged["title"],
                    merged.get("context"), merged.get("instructions", ""),
                    source_pages, 1,
                )
                inserted = insert_questions(conn, exercise_id, merged["questions"], source_pages)
                total_inserted += inserted
                print(f"  [{merged['title']}] Inserted: {inserted}/{len(merged['questions'])} questions")
            except Exception as e:
                print(f"  ERROR: DB insert failed: {e}")
                conn.rollback()
                total_errors += 1

        else:
            # ----- Loose image: each page is its own exercise(s) -----
            for custom_id, meta in pages:
                img_path = meta["image_path"]
                print(f"--- {img_path.name} ---")
                result = llm_results.get(custom_id)
                if result is None or "_error" in result:
                    print(f"  ERROR: {result}")
                    total_errors += 1
                    continue

                print(f"  [LLM RAW RESPONSE]\n{json.dumps(result, indent=2)}\n")

                questions = result.get("questions", [])
                exercises = result.get("exercises", [])
                page_topic = result.get("topic", "general")

                if not exercises and questions:
                    exercises = [{"title": page_topic, "instructions": "", "questions": questions}]

                if not exercises:
                    print(f"  No exercises extracted.")
                    total_errors += 1
                    continue

                total_q = sum(len(ex.get("questions", [])) for ex in exercises)
                print(f"  Extracted: {len(exercises)} exercise(s), {total_q} questions | Topic: {page_topic}")

                if dry_run:
                    for ei, ex in enumerate(exercises, 1):
                        print(f"  Exercise {ei}: {ex.get('title', '')}")
                        if ex.get("context"):
                            print(f"    Context: {ex['context'][:120]}…")
                        if ex.get("instructions"):
                            print(f"    Instructions: {ex['instructions'][:80]}")
                        for j, q in enumerate(ex.get("questions", []), 1):
                            print(f"    Q{j}: {q.get('question', '')[:80]}…")
                            print(f"    A{j}: {q.get('answer', '')[:80]}")
                    continue

                page_topic_id = upsert_topic(conn, subject_id, page_topic)
                source_page = img_path.stem
                try:
                    for ei, ex in enumerate(exercises):
                        ex_questions = ex.get("questions", [])
                        if not ex_questions:
                            continue
                        exercise_id = insert_exercise(
                            conn, page_topic_id,
                            ex.get("title", page_topic),
                            ex.get("context") or None,
                            ex.get("instructions", ""),
                            source_page, ei + 1,
                        )
                        inserted = insert_questions(conn, exercise_id, ex_questions, source_page)
                        total_inserted += inserted
                        print(f"  [{ex.get('title', page_topic)}] Inserted: {inserted}/{len(ex_questions)} questions")
                except Exception as e:
                    print(f"  ERROR: DB insert failed: {e}")
                    conn.rollback()
                    total_errors += 1

    return total_inserted, total_errors


def ingest_folder(config: dict, folder: Path, subject: str, dry_run: bool, immediate: bool = False, resume_batch_id: str | None = None):
    topics = discover_topics(folder)
    if not topics:
        print(f"No images found in {folder}")
        return

    registry = build_image_registry(topics)
    total_images = len(registry)

    subfolder_count = sum(1 for m in registry.values() if m["should_merge"])
    loose_count = total_images - subfolder_count

    print(f"Found {total_images} images in {folder}")
    print(f"Subject: {subject}")
    if subfolder_count:
        print(f"Subfolder images (AI auto-detect topic, merge pages): {subfolder_count}")
    if loose_count:
        print(f"Loose images (AI auto-detect topic per page): {loose_count}")
    print(f"Mode: {'immediate' if immediate else 'batch'} | Dry run: {dry_run}")
    print()

    # ------------------------------------------------------------------ #
    # Phase 2: Obtain LLM results — batch or immediate                    #
    # ------------------------------------------------------------------ #
    llm_cfg = config["llm"]
    api_key = os.getenv("INGEST_LLM_API_KEY") or llm_cfg.get("api_key", "")
    batch_base_url = get_batch_base_url(config)

    if immediate:
        # Original sequential behaviour
        llm_results: dict = {}
        for i, (custom_id, meta) in enumerate(registry.items(), 1):
            print(f"[{i}/{total_images}] Processing: {meta['image_path'].name}")
            try:
                result = call_llm(config, str(meta["image_path"]), subject, meta["topic"])
                llm_results[custom_id] = result
            except Exception as e:
                print(f"  ERROR: LLM call failed: {e}")
                llm_results[custom_id] = {"_error": str(e)}

            if i < total_images:
                delay = llm_cfg.get("delay_between_calls", 2)
                time.sleep(delay)
    else:
        # Batch API mode
        if resume_batch_id:
            print(f"Resuming batch {resume_batch_id} …")
            batch_id = resume_batch_id
        else:
            print("Building batch JSONL …")
            jsonl = build_batch_jsonl(config, registry, subject)
            file_id = upload_batch_file(batch_base_url, api_key, jsonl)
            batch_id = create_batch(batch_base_url, api_key, file_id)

        poll_interval = llm_cfg.get("batch_poll_interval", 30)
        output_file_id = poll_batch(batch_base_url, api_key, batch_id, poll_interval=poll_interval)
        llm_results = download_batch_results(batch_base_url, api_key, output_file_id)

    print()

    # ------------------------------------------------------------------ #
    # Phase 3: Process results and write to DB                            #
    # ------------------------------------------------------------------ #
    conn = None
    subject_id = None
    if not dry_run:
        conn = get_db_connection(config)
        ensure_schema(conn)
        subject_id = upsert_subject(conn, subject)

    total_inserted, total_errors = _process_all_results(
        llm_results, registry, conn, subject_id, dry_run
    )

    if conn:
        conn.close()

    print()
    print(f"Done. Total inserted: {total_inserted}, Errors: {total_errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest textbook photos into curriculum database")
    parser.add_argument("--folder", "-f", type=str, default=None, help="Folder containing page photos organized in subfolders by topic")
    parser.add_argument("--subject", "-s", type=str, default="general", help="Subject name (e.g., math, science, ela)")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Extract questions but don't insert into DB")
    parser.add_argument("--immediate", action="store_true", help="Call LLM one image at a time (skips Batch API; slower, no discount)")
    parser.add_argument("--resume-batch", metavar="BATCH_ID", default=None, help="Resume polling an existing batch job by its ID")
    args = parser.parse_args()

    config = load_config()

    folder = Path(args.folder) if args.folder else Path(config.get("default_folder", "./pages"))
    if not folder.exists():
        print(f"ERROR: Folder {folder} does not exist")
        sys.exit(1)

    ingest_folder(
        config, folder, args.subject, args.dry_run,
        immediate=args.immediate,
        resume_batch_id=args.resume_batch,
    )


if __name__ == "__main__":
    main()