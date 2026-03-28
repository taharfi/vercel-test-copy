from threading import Lock, Thread
import re
from typing import Dict, List

from flask import Flask, redirect, render_template, request, url_for
from google import genai

from batch_agent import (
    build_wp_auth_header,
    fetch_wordpress_categories,
    get_default_prompt_template,
    normalize_gemini_api_keys,
    parse_manga_line,
    run_batch_entries,
)


app = Flask(__name__)

state_lock = Lock()
app_state: Dict[str, object] = {
    "running": False,
    "logs": [],
    "results": [],
    "form": {
        "wordpress_base_url": "",
        "wordpress_username": "",
        "wordpress_app_password": "",
        "gemini_api_key": "",
        "gemini_api_keys": "",
        "wp_admin_username": "",
        "wp_admin_password": "",
        "category_thumbnail_meta_key": "category_thumbnail_id",
        "prompt_templates": get_default_prompt_template(),
        "sleep_seconds": "60",
    },
    "categories": [],
}


def append_log(message: str) -> None:
    with state_lock:
        logs: List[str] = app_state["logs"]  # type: ignore[assignment]
        logs.append(message)
        if len(logs) > 500:
            del logs[:-500]


def set_results(results: List[Dict[str, str]]) -> None:
    with state_lock:
        app_state["results"] = results
        app_state["running"] = False


def set_running(value: bool) -> None:
    with state_lock:
        app_state["running"] = value


def persist_form(form: Dict[str, str]) -> None:
    with state_lock:
        app_state["form"] = form


def parse_sleep_seconds(value: str) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 60


def parse_prompt_templates(value: str) -> List[str]:
    chunks = [chunk.strip() for chunk in re.split(r"^\s*---\s*$", value, flags=re.MULTILINE)]
    cleaned = [chunk for chunk in chunks if chunk]
    return cleaned or [get_default_prompt_template()]


def parse_gemini_api_keys(primary_key: str, bulk_value: str) -> List[str]:
    extra_keys = [line.strip() for line in bulk_value.splitlines() if line.strip()]
    return normalize_gemini_api_keys(primary_key, extra_keys)


def worker_entries(
    wordpress_base_url: str,
    wordpress_username: str,
    wordpress_app_password: str,
    gemini_api_keys: List[str],
    wp_admin_username: str,
    wp_admin_password: str,
    category_thumbnail_meta_key: str,
    prompt_templates: List[str],
    sleep_seconds: int,
    entries: List[tuple[int, str, str]],
) -> None:
    try:
        append_log(f"[INFO] Starting batch for {len(entries)} selected manga.")
        auth_headers = build_wp_auth_header(wordpress_username, wordpress_app_password)
        clients = [genai.Client(api_key=api_key) for api_key in gemini_api_keys]
        results = run_batch_entries(
            gemini_clients=clients,
            wordpress_base_url=wordpress_base_url,
            auth_headers=auth_headers,
            entries=entries,
            prompt_templates=prompt_templates,
            wp_admin_username=wp_admin_username,
            wp_admin_password=wp_admin_password,
            category_thumbnail_meta_key=category_thumbnail_meta_key,
            sleep_seconds=sleep_seconds,
            logger=append_log,
        )
        set_results(results)
    except Exception as exc:
        append_log(f"[FATAL] {exc}")
        set_results([])


@app.get("/")
def index():
    with state_lock:
        context = {
            "running": app_state["running"],
            "logs": list(app_state["logs"]),  # type: ignore[arg-type]
            "results": list(app_state["results"]),  # type: ignore[arg-type]
            "form": dict(app_state["form"]),  # type: ignore[arg-type]
            "categories": list(app_state["categories"]),  # type: ignore[arg-type]
        }
    return render_template("index.html", **context)


@app.post("/fetch-categories")
def fetch_categories():
    form = {
        "wordpress_base_url": request.form.get("wordpress_base_url", "").strip(),
        "wordpress_username": request.form.get("wordpress_username", "").strip(),
        "wordpress_app_password": request.form.get("wordpress_app_password", "").strip(),
        "gemini_api_key": request.form.get("gemini_api_key", "").strip(),
        "gemini_api_keys": request.form.get("gemini_api_keys", "").strip(),
        "wp_admin_username": request.form.get("wp_admin_username", "").strip(),
        "wp_admin_password": request.form.get("wp_admin_password", "").strip(),
        "category_thumbnail_meta_key": request.form.get("category_thumbnail_meta_key", "category_thumbnail_id").strip() or "category_thumbnail_id",
        "prompt_templates": request.form.get("prompt_templates", "").strip() or get_default_prompt_template(),
        "sleep_seconds": request.form.get("sleep_seconds", "60").strip(),
    }
    persist_form(form)

    if not all([form["wordpress_base_url"], form["wordpress_username"], form["wordpress_app_password"]]):
        append_log("[ERROR] WordPress URL, username, and application password are required.")
        return redirect(url_for("index"))

    try:
        auth_headers = build_wp_auth_header(form["wordpress_username"], form["wordpress_app_password"])
        categories = fetch_wordpress_categories(form["wordpress_base_url"], auth_headers)
        prepared = []
        for category in categories:
            raw_name = str(category.get("name", "")).strip()
            if not raw_name:
                continue

            category_id, clean_name, language = parse_manga_line(f"{category['id']}, {raw_name}")
            prepared.append(
                {
                    "id": category_id,
                    "raw_name": raw_name,
                    "clean_name": clean_name,
                    "language": language,
                    "description": str(category.get("description", "") or ""),
                    "selected": False,
                }
            )

        with state_lock:
            app_state["categories"] = prepared

        append_log(f"[INFO] Fetched {len(prepared)} categories from WordPress.")
    except Exception as exc:
        append_log(f"[ERROR] Failed to fetch categories: {exc}")

    return redirect(url_for("index"))


@app.post("/run")
def run():
    form = {
        "wordpress_base_url": request.form.get("wordpress_base_url", "").strip(),
        "wordpress_username": request.form.get("wordpress_username", "").strip(),
        "wordpress_app_password": request.form.get("wordpress_app_password", "").strip(),
        "gemini_api_key": request.form.get("gemini_api_key", "").strip(),
        "gemini_api_keys": request.form.get("gemini_api_keys", "").strip(),
        "wp_admin_username": request.form.get("wp_admin_username", "").strip(),
        "wp_admin_password": request.form.get("wp_admin_password", "").strip(),
        "category_thumbnail_meta_key": request.form.get("category_thumbnail_meta_key", "category_thumbnail_id").strip() or "category_thumbnail_id",
        "prompt_templates": request.form.get("prompt_templates", "").strip() or get_default_prompt_template(),
        "sleep_seconds": request.form.get("sleep_seconds", "60").strip(),
    }
    persist_form(form)

    if not all(
        [
            form["wordpress_base_url"],
            form["wordpress_username"],
            form["wordpress_app_password"],
            form["gemini_api_key"],
        ]
    ):
        append_log("[ERROR] All connection fields are required before starting the batch.")
        return redirect(url_for("index"))

    selected_ids = {int(value) for value in request.form.getlist("selected_category_ids")}

    with state_lock:
        categories = list(app_state["categories"])  # type: ignore[arg-type]
        if app_state["running"]:
            append_log("[WARN] A batch is already running.")
            return redirect(url_for("index"))
        app_state["running"] = True
        app_state["logs"] = []
        app_state["results"] = []
        app_state["categories"] = [
            {
                **category,
                "selected": category["id"] in selected_ids,
            }
            for category in categories
        ]

    selected_entries = [
        (int(category["id"]), str(category["clean_name"]), str(category["language"]))
        for category in categories
        if int(category["id"]) in selected_ids
    ]

    if not selected_entries:
        set_running(False)
        append_log("[ERROR] Select at least one category before starting the batch.")
        return redirect(url_for("index"))

    Thread(
        target=worker_entries,
        args=(
            form["wordpress_base_url"],
            form["wordpress_username"],
            form["wordpress_app_password"],
            parse_gemini_api_keys(form["gemini_api_key"], form["gemini_api_keys"]),
            form["wp_admin_username"],
            form["wp_admin_password"],
            form["category_thumbnail_meta_key"],
            parse_prompt_templates(form["prompt_templates"]),
            parse_sleep_seconds(form["sleep_seconds"]),
            selected_entries,
        ),
        daemon=True,
    ).start()
    append_log(f"[INFO] Queued {len(selected_entries)} selected manga.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
