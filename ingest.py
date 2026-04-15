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

Config:
  Copy config.example.yaml to config.yaml and customize prompts + DB settings.
  Set INGEST_LLM_API_KEY env var (or put it in config.yaml).
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
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


def call_llm(config: dict, image_path: str, subject: str, topic: str | None) -> dict:
    """Send image to multimodal LLM and get structured Q&A JSON back."""
    llm_cfg = config["llm"]
    api_key = os.getenv("INGEST_LLM_API_KEY") or llm_cfg.get("api_key", "")
    base_url = llm_cfg["base_url"].rstrip("/")
    model = llm_cfg["model"]

    b64 = encode_image(image_path)
    mime = get_image_mime(image_path)

    system_prompt = config["prompts"]["system"]
    user_prompt = config["prompts"]["user"]

    # Replace template variables
    user_prompt = user_prompt.replace("{{SUBJECT}}", subject)
    user_prompt = user_prompt.replace("{{TOPIC}}", topic or "auto-detect")
    user_prompt = user_prompt.replace("{{FILENAME}}", Path(image_path).name)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                        },
                    },
                ],
            },
        ],
        "temperature": llm_cfg.get("temperature", 0.1),
        "max_tokens": llm_cfg.get("max_tokens", 4096),
        "response_format": {"type": "json_object"},
    }

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


def _merge_subfolder_pages(
    config: dict, images: list[Path], subject: str,
    dry_run: bool, total_images: int, img_num: int, total_errors: int,
) -> dict:
    """Process all pages in a subfolder and merge into a single exercise.

    For multi-page exercises, the first page typically has the context/passage
    and subsequent pages have questions that reference it. We merge all pages
    into one exercise with one context and all questions combined.

    The topic is auto-detected by the AI from the page images.
    """
    merged_context = None
    merged_instructions = ""
    merged_title = None
    merged_topic = None  # Auto-detected by AI from page images
    merged_questions: list[dict] = []

    for img_path in images:
        img_num += 1
        print(f"[{img_num}/{total_images}] Processing: {img_path.name}")

        try:
            result = call_llm(config, str(img_path), subject, None)  # None = AI auto-detect
        except Exception as e:
            print(f"  ERROR: LLM call failed: {e}")
            total_errors += 1
            continue

        # Capture topic from the first LLM response that has one
        if not merged_topic:
            merged_topic = result.get("topic", None)

        exercises = result.get("exercises", [])
        questions = result.get("questions", [])

        # Support flat questions list (legacy)
        if not exercises and questions:
            fallback_title = merged_topic or "general"
            exercises = [{"title": fallback_title, "instructions": "", "questions": questions}]

        for ex in exercises:
            ex_questions = ex.get("questions", [])
            if not ex_questions:
                continue

            # Use context from the first page that has one
            if merged_context is None and ex.get("context"):
                merged_context = ex["context"]

            # Use instructions from the first page that has them
            if not merged_instructions and ex.get("instructions"):
                merged_instructions = ex["instructions"]

            # Use title from the first exercise that has one
            if not merged_title and ex.get("title"):
                merged_title = ex["title"]

            merged_questions.extend(ex_questions)

        total_q = sum(len(ex.get("questions", [])) for ex in exercises) if exercises else 0
        print(f"  Extracted: {len(exercises)} exercise(s), {total_q} questions | Topic: {merged_topic or 'auto-detect'}")

        # Rate limit between images
        if img_num < total_images:
            delay = config["llm"].get("delay_between_calls", 2)
            time.sleep(delay)

    # Fallbacks for topic and title
    if not merged_topic:
        merged_topic = "general"
    if not merged_title:
        merged_title = merged_topic

    return {
        "topic": merged_topic,
        "title": merged_title,
        "context": merged_context,
        "instructions": merged_instructions,
        "questions": merged_questions,
        "img_num": img_num,
        "total_errors": total_errors,
    }


def ingest_folder(config: dict, folder: Path, subject: str, dry_run: bool):
    topics = discover_topics(folder)
    if not topics:
        print(f"No images found in {folder}")
        return

    total_images = sum(len(imgs) for _, imgs, _ in topics)
    subfolder_count = sum(len(imgs) for _, imgs, merge in topics if merge)
    auto_count = sum(len(imgs) for _, imgs, merge in topics if not merge)
    print(f"Found {total_images} images in {folder}")
    print(f"Subject: {subject}")
    if subfolder_count:
        print(f"Subfolder images (AI auto-detect topic, merge pages): {subfolder_count}")
    if auto_count:
        print(f"Loose images (AI auto-detect topic per page): {auto_count}")
    print(f"Dry run: {dry_run}")
    print()

    conn = None
    if not dry_run:
        conn = get_db_connection(config)
        ensure_schema(conn)
        subject_id = upsert_subject(conn, subject)

    total_inserted = 0
    total_errors = 0
    img_num = 0

    for topic_name, images, should_merge in topics:
        if should_merge:
            print(f"--- Subfolder ({len(images)} pages, AI auto-detect topic) ---")
        else:
            print(f"--- Loose images ({len(images)} pages, AI auto-detect) ---")

        # For subfolders, we merge all pages into a single exercise.
        # Topic is auto-detected by AI from page images.
        if should_merge:
            merged_exercise = _merge_subfolder_pages(
                config, images, subject, dry_run, total_images, img_num, total_errors
            )
            # Update img_num and total_errors from the call
            img_num = merged_exercise["img_num"]
            total_errors = merged_exercise["total_errors"]

            detected_topic = merged_exercise["topic"]

            if not merged_exercise["questions"]:
                print(f"  No questions extracted across all pages.")
                total_errors += 1
                continue

            if dry_run:
                ex = merged_exercise
                print(f"  Merged exercise: {ex['title']} (Topic: {detected_topic})")
                if ex.get('context'):
                    print(f"    Context: {ex['context'][:120]}...")
                if ex.get('instructions'):
                    print(f"    Instructions: {ex['instructions'][:80]}")
                for j, q in enumerate(ex["questions"], 1):
                    print(f"    Q{j}: {q.get('question', '')[:80]}...")
                    print(f"    A{j}: {q.get('answer', '')[:80]}")
                continue

            # Resolve topic_id for the AI-detected topic
            topic_id = upsert_topic(conn, subject_id, detected_topic)
            source_pages = "+".join(img.stem for img in images)
            try:
                exercise_id = insert_exercise(
                    conn, topic_id, merged_exercise["title"],
                    merged_exercise.get("context"), merged_exercise.get("instructions", ""),
                    source_pages, 1
                )
                inserted = insert_questions(conn, exercise_id, merged_exercise["questions"], source_pages)
                total_inserted += inserted
                print(f"  [{merged_exercise['title']}] Inserted: {inserted}/{len(merged_exercise['questions'])} questions")
            except Exception as e:
                print(f"  ERROR: DB insert failed: {e}")
                if conn:
                    conn.rollback()
                total_errors += 1

        else:
            # Loose images: each page is its own exercise (LLM auto-detects topic)
            for img_path in images:
                img_num += 1
                print(f"[{img_num}/{total_images}] Processing: {img_path.name}")

                try:
                    result = call_llm(config, str(img_path), subject, topic_name)
                except Exception as e:
                    print(f"  ERROR: LLM call failed: {e}")
                    total_errors += 1
                    continue

                questions = result.get("questions", [])
                exercises = result.get("exercises", [])

                # LLM returns the detected topic
                page_topic = result.get("topic", "general")

                # Support flat questions list (legacy) by wrapping in a single exercise
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
                        if ex.get('context'):
                            print(f"    Context: {ex['context'][:120]}...")
                        if ex.get('instructions'):
                            print(f"    Instructions: {ex['instructions'][:80]}")
                        for j, q in enumerate(ex.get("questions", []), 1):
                            print(f"    Q{j}: {q.get('question', '')[:80]}...")
                            print(f"    A{j}: {q.get('answer', '')[:80]}")
                    continue

                page_topic_id = upsert_topic(conn, subject_id, page_topic)

                source_page = img_path.stem
                try:
                    for ei, ex in enumerate(exercises):
                        ex_title = ex.get("title", page_topic)
                        ex_context = ex.get("context") or None
                        ex_instructions = ex.get("instructions", "")
                        ex_questions = ex.get("questions", [])
                        if not ex_questions:
                            continue
                        exercise_id = insert_exercise(conn, page_topic_id, ex_title, ex_context, ex_instructions, source_page, ei + 1)
                        inserted = insert_questions(conn, exercise_id, ex_questions, source_page)
                        total_inserted += inserted
                        print(f"  [{ex_title}] Inserted: {inserted}/{len(ex_questions)} questions")
                except Exception as e:
                    print(f"  ERROR: DB insert failed: {e}")
                    if conn:
                        conn.rollback()
                    total_errors += 1
                    continue

                # Rate limit between images
                if img_num < total_images:
                    delay = config["llm"].get("delay_between_calls", 2)
                    time.sleep(delay)

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
    args = parser.parse_args()

    config = load_config()

    folder = Path(args.folder) if args.folder else Path(config.get("default_folder", "./pages"))
    if not folder.exists():
        print(f"ERROR: Folder {folder} does not exist")
        sys.exit(1)

    ingest_folder(config, folder, args.subject, args.dry_run)


if __name__ == "__main__":
    main()