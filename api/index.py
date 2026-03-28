import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, render_template, request
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from batch_agent import (
    build_wp_auth_header,
    fetch_wordpress_categories,
    get_default_prompt_template,
    normalize_gemini_api_keys,
    parse_manga_line,
    process_manga,
)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static",
)


def default_form() -> Dict[str, str]:
    return {
        "wordpress_base_url": "",
        "wordpress_username": "",
        "wordpress_app_password": "",
        "gemini_api_key": "",
        "gemini_api_keys": "",
        "wp_admin_username": "",
        "wp_admin_password": "",
        "category_thumbnail_meta_key": "category_thumbnail_id",
        "prompt_templates": get_default_prompt_template(),
    }


def parse_prompt_templates(value: str) -> List[str]:
    chunks = [chunk.strip() for chunk in re.split(r"^\s*---\s*$", value, flags=re.MULTILINE)]
    cleaned = [chunk for chunk in chunks if chunk]
    return cleaned or [get_default_prompt_template()]


def parse_categories(raw_value: str) -> List[Dict[str, Any]]:
    if not raw_value.strip():
        return []
    try:
        parsed = json.loads(raw_value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def parse_results(raw_value: str) -> List[Dict[str, str]]:
    if not raw_value.strip():
        return []
    try:
        parsed = json.loads(raw_value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def render_page(
    *,
    form: Dict[str, str],
    categories: List[Dict[str, Any]],
    results: List[Dict[str, str]],
    logs: List[str],
    message: str = "",
) -> str:
    return render_template(
        "vercel_index.html",
        form=form,
        categories=categories,
        categories_json=json.dumps(categories),
        results=results,
        results_json=json.dumps(results),
        logs=logs,
        message=message,
    )


@app.get("/")
def index():
    return render_page(form=default_form(), categories=[], results=[], logs=[])


@app.post("/fetch-categories")
def fetch_categories():
    form = {
        **default_form(),
        "wordpress_base_url": request.form.get("wordpress_base_url", "").strip(),
        "wordpress_username": request.form.get("wordpress_username", "").strip(),
        "wordpress_app_password": request.form.get("wordpress_app_password", "").strip(),
        "gemini_api_key": request.form.get("gemini_api_key", "").strip(),
        "gemini_api_keys": request.form.get("gemini_api_keys", "").strip(),
        "wp_admin_username": request.form.get("wp_admin_username", "").strip(),
        "wp_admin_password": request.form.get("wp_admin_password", "").strip(),
        "category_thumbnail_meta_key": request.form.get("category_thumbnail_meta_key", "category_thumbnail_id").strip()
        or "category_thumbnail_id",
        "prompt_templates": request.form.get("prompt_templates", "").strip() or get_default_prompt_template(),
    }

    logs: List[str] = []
    results = parse_results(request.form.get("results_json", ""))

    if not all([form["wordpress_base_url"], form["wordpress_username"], form["wordpress_app_password"]]):
        logs.append("[ERROR] WordPress URL, username, and application password are required.")
        return render_page(form=form, categories=[], results=results, logs=logs)

    try:
        auth_headers = build_wp_auth_header(form["wordpress_username"], form["wordpress_app_password"])
        fetched = fetch_wordpress_categories(form["wordpress_base_url"], auth_headers)
        categories: List[Dict[str, Any]] = []
        for category in fetched:
            raw_name = str(category.get("name", "")).strip()
            if not raw_name:
                continue

            category_id, clean_name, language = parse_manga_line(f"{category['id']}, {raw_name}")
            categories.append(
                {
                    "id": category_id,
                    "raw_name": raw_name,
                    "clean_name": clean_name,
                    "language": language,
                    "description": str(category.get("description", "") or ""),
                    "selected": False,
                }
            )
        logs.append(f"[INFO] Fetched {len(categories)} categories from WordPress.")
        return render_page(form=form, categories=categories, results=results, logs=logs)
    except Exception as exc:
        logs.append(f"[ERROR] Failed to fetch categories: {exc}")
        return render_page(form=form, categories=[], results=results, logs=logs)


@app.post("/process-selected")
def process_selected():
    form = {
        **default_form(),
        "wordpress_base_url": request.form.get("wordpress_base_url", "").strip(),
        "wordpress_username": request.form.get("wordpress_username", "").strip(),
        "wordpress_app_password": request.form.get("wordpress_app_password", "").strip(),
        "gemini_api_key": request.form.get("gemini_api_key", "").strip(),
        "gemini_api_keys": request.form.get("gemini_api_keys", "").strip(),
        "wp_admin_username": request.form.get("wp_admin_username", "").strip(),
        "wp_admin_password": request.form.get("wp_admin_password", "").strip(),
        "category_thumbnail_meta_key": request.form.get("category_thumbnail_meta_key", "category_thumbnail_id").strip()
        or "category_thumbnail_id",
        "prompt_templates": request.form.get("prompt_templates", "").strip() or get_default_prompt_template(),
    }

    categories = parse_categories(request.form.get("categories_json", ""))
    results = parse_results(request.form.get("results_json", ""))
    logs: List[str] = []

    selected_ids = {int(value) for value in request.form.getlist("selected_category_ids")}
    if not selected_ids:
        logs.append("[ERROR] Select at least one manga.")
        return render_page(form=form, categories=categories, results=results, logs=logs)

    selected_categories = [item for item in categories if int(item["id"]) in selected_ids]
    if not selected_categories:
        logs.append("[ERROR] Selected manga were not found in the current category payload.")
        return render_page(form=form, categories=categories, results=results, logs=logs)

    current = selected_categories[0]
    prompt_templates = parse_prompt_templates(form["prompt_templates"])
    api_keys = normalize_gemini_api_keys(
        form["gemini_api_key"],
        [line.strip() for line in form["gemini_api_keys"].splitlines() if line.strip()],
    )
    clients = [genai.Client(api_key=api_key) for api_key in api_keys]
    auth_headers = build_wp_auth_header(form["wordpress_username"], form["wordpress_app_password"])

    def log(message: str) -> None:
        logs.append(message)

    try:
        prompt_template = prompt_templates[len(results) % len(prompt_templates)]
        process_manga(
            gemini_clients=clients,
            wordpress_base_url=form["wordpress_base_url"],
            auth_headers=auth_headers,
            category_id=int(current["id"]),
            clean_name=str(current["clean_name"]),
            language=str(current["language"]),
            prompt_template=prompt_template,
            manga_index=len(results),
            wp_admin_username=form["wp_admin_username"],
            wp_admin_password=form["wp_admin_password"],
            category_thumbnail_meta_key=form["category_thumbnail_meta_key"],
            logger=log,
        )
        results.append(
            {
                "category_id": str(current["id"]),
                "title": str(current["clean_name"]),
                "language": str(current["language"]),
                "status": "success",
            }
        )
        categories = [item for item in categories if int(item["id"]) != int(current["id"])]
        message = f"Processed 1 manga successfully. Re-run to continue with the next selected manga."
    except Exception as exc:
        results.append(
            {
                "category_id": str(current["id"]),
                "title": str(current["clean_name"]),
                "language": str(current["language"]),
                "status": "failed",
                "error": str(exc),
            }
        )
        message = f"Processing failed for {current['clean_name']}."
        logs.append(f"[ERROR] {exc}")

    return render_page(form=form, categories=categories, results=results, logs=logs, message=message)


application = app
