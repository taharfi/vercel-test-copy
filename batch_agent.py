import base64
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests
from google import genai
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


MANGAS_FILE = Path("mangas.txt")
TEMP_IMAGE_PATH = Path("temp_banner.webp")
DEBUG_DIR = Path("debug")
JIKAN_SEARCH_URL = "https://api.jikan.moe/v4/manga"
WP_MEDIA_ENDPOINT = "/wp-json/wp/v2/media"
WP_CATEGORIES_LIST_ENDPOINT = "/wp-json/wp/v2/categories"
WP_CATEGORIES_ENDPOINT = "/wp-json/wp/v2/categories/{category_id}"
RATE_LIMIT_SLEEP_SECONDS = 60
MAX_GEMINI_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 60

LANGUAGE_MAP: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "pt": "Portuguese",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "ar": "Arabic",
    "id": "Indonesian",
    "tr": "Turkish",
}

LANGUAGE_TAG_PATTERN = re.compile(r"\s*\(([a-z]{2})\)\s*$", re.IGNORECASE)


@dataclass
class BatchConfig:
    wordpress_base_url: str
    wordpress_username: str
    wordpress_app_password: str
    gemini_api_key: str
    gemini_api_keys: Optional[List[str]] = None
    wp_admin_username: str = ""
    wp_admin_password: str = ""
    category_thumbnail_meta_key: str = "category_thumbnail_id"
    prompt_templates: Optional[List[str]] = None
    sleep_seconds: int = RATE_LIMIT_SLEEP_SECONDS
    mangas_file: Path = MANGAS_FILE


LogCallback = Callable[[str], None]


def load_env(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_wp_auth_header(username: str, application_password: str) -> Dict[str, str]:
    token = base64.b64encode(f"{username}:{application_password}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}


def default_logger(message: str) -> None:
    print(message)


def slugify_filename(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "manga-banner"


def get_default_prompt_template() -> str:
    return (
        "Write a 3-4 sentence, highly engaging, spoiler-free, SEO-optimized summary for the manga '{title}'. "
        "Weave alternate names naturally into the text when relevant. "
        "Alternate names: {alternate_names}. "
        "Write the entire description in {language}. "
        "Keep it readable, natural, and suitable for a WordPress category description."
    )


def normalize_gemini_api_keys(primary_key: str, extra_keys: Optional[List[str]] = None) -> List[str]:
    keys = [primary_key, *(extra_keys or [])]
    cleaned: List[str] = []
    seen = set()

    for key in keys:
        normalized = (key or "").strip()
        if normalized and normalized not in seen:
            cleaned.append(normalized)
            seen.add(normalized)

    if not cleaned:
        raise RuntimeError("At least one Gemini API key is required.")

    return cleaned


def choose_prompt_template(prompt_templates: Optional[List[str]], index: int) -> str:
    templates = [item.strip() for item in (prompt_templates or []) if item and item.strip()]
    if not templates:
        return get_default_prompt_template()
    return templates[index % len(templates)]


def build_prompt(template: str, clean_name: str, target_language: str, alternate_names: List[str]) -> str:
    alternate_text = ", ".join(alternate_names[:6]) if alternate_names else "None provided"
    return template.format(
        title=clean_name,
        language=target_language,
        alternate_names=alternate_text,
    )


def parse_manga_line(line: str) -> Tuple[int, str, str]:
    raw = line.strip()
    if not raw:
        raise ValueError("Encountered an empty line in mangas.txt")

    parts = raw.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid line format: {raw!r}. Expected 'CategoryID, Manga Name'")

    category_id = int(parts[0].strip())
    original_name = parts[1].strip()

    match = LANGUAGE_TAG_PATTERN.search(original_name)
    if match:
        language_code = match.group(1).lower()
        language = LANGUAGE_MAP.get(language_code, "English")
        clean_name = LANGUAGE_TAG_PATTERN.sub("", original_name).strip()
    else:
        language = "English"
        clean_name = original_name

    return category_id, clean_name, language


def read_manga_entries(path: Path) -> List[Tuple[int, str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find input file: {path}")

    entries: List[Tuple[int, str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(parse_manga_line(line))
        except Exception as exc:
            raise ValueError(f"Failed to parse line {line_number}: {exc}") from exc
    return entries


def search_jikan(clean_name: str) -> Dict[str, object]:
    response = requests.get(
        JIKAN_SEARCH_URL,
        params={"q": clean_name, "limit": 1},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data") or []
    if not data:
        raise RuntimeError(f"No manga found on Jikan for '{clean_name}'")
    return data[0]


def fetch_wordpress_categories(
    wordpress_base_url: str,
    auth_headers: Dict[str, str],
) -> List[Dict[str, object]]:
    categories: List[Dict[str, object]] = []
    page = 1

    while True:
        response = requests.get(
            f"{wordpress_base_url.rstrip('/')}{WP_CATEGORIES_LIST_ENDPOINT}",
            headers=auth_headers,
            params={"per_page": 100, "page": page},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            break

        categories.extend(payload)
        if len(payload) < 100:
            break
        page += 1

    return categories


def extract_cover_image_url(jikan_manga: Dict[str, object]) -> str:
    images = jikan_manga.get("images") or {}
    webp = images.get("webp") or {}
    image_url = webp.get("large_image_url")
    if not image_url:
        raise RuntimeError("Jikan response did not include images.webp.large_image_url")
    return image_url


def extract_alternate_names(jikan_manga: Dict[str, object]) -> List[str]:
    names: List[str] = []
    for item in jikan_manga.get("titles") or []:
        title = item.get("title")
        if title:
            names.append(str(title))

    for key in ("title", "title_english", "title_japanese"):
        value = jikan_manga.get(key)
        if value:
            names.append(str(value))

    deduped: List[str] = []
    seen = set()
    for name in names:
        normalized = name.strip()
        lowered = normalized.lower()
        if normalized and lowered not in seen:
            deduped.append(normalized)
            seen.add(lowered)
    return deduped


def is_retryable_gemini_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 503):
        return True

    code = getattr(exc, "code", None)
    if code in (429, 503):
        return True

    message = str(exc).lower()
    return "429" in message or "503" in message or "rate limit" in message or "server unavailable" in message


def generate_description(
    clients: List[genai.Client],
    clean_name: str,
    target_language: str,
    alternate_names: List[str],
    prompt_template: str,
    manga_index: int,
    logger: LogCallback = default_logger,
) -> str:
    prompt = build_prompt(prompt_template, clean_name, target_language, alternate_names)

    backoff = INITIAL_BACKOFF_SECONDS
    for attempt in range(1, MAX_GEMINI_RETRIES + 1):
        try:
            client_index = (manga_index + attempt - 1) % len(clients)
            logger(
                f"[INFO] Gemini attempt {attempt}/{MAX_GEMINI_RETRIES} for '{clean_name}' "
                f"using API key #{client_index + 1}."
            )
            response = clients[client_index].models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = getattr(response, "text", "") or ""
            text = text.strip()
            if not text:
                raise RuntimeError("Gemini returned an empty response")
            return text
        except Exception as exc:
            if attempt >= MAX_GEMINI_RETRIES or not is_retryable_gemini_error(exc):
                raise
            logger(
                f"[WARN] Gemini attempt {attempt} failed with a retryable error: {exc}. "
                f"Switching keys and sleeping {backoff} seconds before retrying."
            )
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("Gemini generation failed after all retries")


def download_image(image_url: str, destination: Path) -> None:
    response = requests.get(image_url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def upload_media(
    wordpress_base_url: str,
    auth_headers: Dict[str, str],
    image_path: Path,
    manga_name: str,
) -> int:
    filename_slug = slugify_filename(manga_name)
    upload_filename = f"{filename_slug}.webp"

    with image_path.open("rb") as handle:
        response = requests.post(
            f"{wordpress_base_url.rstrip('/')}{WP_MEDIA_ENDPOINT}",
            headers={
                **auth_headers,
                "Content-Disposition": f'attachment; filename="{upload_filename}"',
                "Content-Type": "image/webp",
            },
            data=handle.read(),
            timeout=60,
        )
    response.raise_for_status()
    payload = response.json()
    media_id = payload.get("id")
    if not media_id:
        raise RuntimeError("WordPress media upload did not return an ID")

    media_update_response = requests.post(
        f"{wordpress_base_url.rstrip('/')}{WP_MEDIA_ENDPOINT}/{int(media_id)}",
        headers={**auth_headers, "Content-Type": "application/json"},
        json={
            "title": manga_name,
            "alt_text": manga_name,
        },
        timeout=60,
    )
    media_update_response.raise_for_status()
    return int(media_id)


def update_category(
    wordpress_base_url: str,
    auth_headers: Dict[str, str],
    category_id: int,
    description: str,
    media_id: int,
    meta_key: str,
) -> None:
    endpoint = WP_CATEGORIES_ENDPOINT.format(category_id=category_id)
    response = requests.post(
        f"{wordpress_base_url.rstrip('/')}{endpoint}",
        headers={**auth_headers, "Content-Type": "application/json"},
        json={
            "description": description,
            "meta": {
                meta_key: media_id,
            },
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    returned_meta = payload.get("meta") or {}

    if returned_meta.get(meta_key) != media_id:
        raise RuntimeError(
            f"WordPress updated the category response without persisting meta '{meta_key}'. "
            f"Make sure the term meta is registered with show_in_rest=true."
        )


def set_category_thumbnail_via_wp_admin(
    wordpress_base_url: str,
    wp_admin_username: str,
    wp_admin_password: str,
    category_id: int,
    description: str,
    media_id: int,
    logger: LogCallback = default_logger,
) -> None:
    if not wp_admin_username or not wp_admin_password:
        raise RuntimeError("WP admin username/password are required for browser automation fallback.")

    edit_url = (
        f"{wordpress_base_url.rstrip('/')}/wp-admin/term.php"
        f"?taxonomy=category&tag_ID={category_id}&post_type=wp-manga"
    )
    login_url = f"{wordpress_base_url.rstrip('/')}/wp-login.php"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            logger(f"[INFO] Opening wp-admin login for category {category_id}.")
            page.goto(login_url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector("#user_login", state="attached", timeout=60_000)
            page.fill("#user_login", wp_admin_username)
            page.fill("#user_pass", wp_admin_password)
            page.click("#wp-submit")
            page.wait_for_load_state("networkidle", timeout=60_000)

            if "wp-login.php" in page.url.lower():
                login_error = ""
                if page.locator("#login_error").count() > 0:
                    login_error = page.locator("#login_error").inner_text().strip()
                raise RuntimeError(
                    f"wp-admin login did not complete. Current URL: {page.url}. "
                    f"{('Login error: ' + login_error) if login_error else 'Check credentials, 2FA, or security plugins.'}"
                )

            logger(f"[INFO] Opening category edit page for {category_id}.")
            page.goto(edit_url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector('input[name="category_thumbnail_id"]', state="attached", timeout=60_000)
            page.evaluate(
                """
                (payload) => {
                    const textarea = document.querySelector('textarea[name="description"]');
                    if (textarea) {
                        textarea.value = payload.description;
                    }

                    if (window.tinyMCE && window.tinyMCE.get('cat_description')) {
                        window.tinyMCE.get('cat_description').setContent(payload.description);
                    }

                    const input = document.querySelector('input[name="category_thumbnail_id"]');
                    if (!input) {
                        throw new Error('category_thumbnail_id input not found');
                    }
                    input.value = String(payload.mediaId);
                    input.setAttribute('value', String(payload.mediaId));
                }
                """,
                {"mediaId": media_id, "description": description},
            )
            page.evaluate(
                """
                () => {
                    const form = document.querySelector('#edittag');
                    if (!form) {
                        throw new Error('Category edit form #edittag not found');
                    }
                    form.submit();
                }
                """
            )
            page.wait_for_load_state("networkidle", timeout=60_000)

            page.goto(edit_url, wait_until="domcontentloaded", timeout=60_000)
            page.wait_for_selector('input[name="category_thumbnail_id"]', state="attached", timeout=60_000)
            saved_value = page.input_value('input[name="category_thumbnail_id"]')
            if str(saved_value).strip() != str(media_id):
                raise RuntimeError(
                    f"Browser automation submitted the form, but category_thumbnail_id is '{saved_value}' instead of '{media_id}'."
                )

            logger(f"[OK] Browser automation attached media {media_id} to category {category_id}.")
        except PlaywrightTimeoutError as exc:
            DEBUG_DIR.mkdir(exist_ok=True)
            screenshot_path = DEBUG_DIR / f"wp-admin-timeout-category-{category_id}.png"
            html_path = DEBUG_DIR / f"wp-admin-timeout-category-{category_id}.html"
            page.screenshot(path=str(screenshot_path), full_page=True)
            html_path.write_text(page.content(), encoding="utf-8")
            logger(f"[DEBUG] Saved screenshot to {screenshot_path}")
            logger(f"[DEBUG] Saved HTML dump to {html_path}")
            raise RuntimeError(f"wp-admin automation timed out: {exc}") from exc
        except Exception:
            DEBUG_DIR.mkdir(exist_ok=True)
            screenshot_path = DEBUG_DIR / f"wp-admin-error-category-{category_id}.png"
            html_path = DEBUG_DIR / f"wp-admin-error-category-{category_id}.html"
            page.screenshot(path=str(screenshot_path), full_page=True)
            html_path.write_text(page.content(), encoding="utf-8")
            logger(f"[DEBUG] Saved screenshot to {screenshot_path}")
            logger(f"[DEBUG] Saved HTML dump to {html_path}")
            raise
        finally:
            browser.close()


def process_manga(
    gemini_clients: List[genai.Client],
    wordpress_base_url: str,
    auth_headers: Dict[str, str],
    category_id: int,
    clean_name: str,
    language: str,
    prompt_template: str,
    manga_index: int,
    wp_admin_username: str,
    wp_admin_password: str,
    category_thumbnail_meta_key: str,
    logger: LogCallback = default_logger,
) -> None:
    logger(f"[INFO] Processing category {category_id}: {clean_name} [{language}]")
    logger(f"[INFO] Searching Jikan for '{clean_name}'.")
    jikan_manga = search_jikan(clean_name)
    logger(f"[INFO] Jikan match found for '{clean_name}'.")
    image_url = extract_cover_image_url(jikan_manga)
    alternate_names = extract_alternate_names(jikan_manga)
    logger(f"[INFO] Generating description for '{clean_name}' with Gemini.")
    description = generate_description(
        gemini_clients,
        clean_name,
        language,
        alternate_names,
        prompt_template,
        manga_index,
        logger=logger,
    )
    logger(f"[INFO] Gemini description generated for '{clean_name}'.")
    logger(f"[INFO] Downloading cover image for '{clean_name}'.")
    download_image(image_url, TEMP_IMAGE_PATH)
    logger(f"[INFO] Uploading media to WordPress for '{clean_name}'.")
    media_id = upload_media(wordpress_base_url, auth_headers, TEMP_IMAGE_PATH, clean_name)
    logger(f"[INFO] Media uploaded for '{clean_name}' with ID {media_id}.")
    try:
        logger(f"[INFO] Attempting REST category update for '{clean_name}'.")
        update_category(
            wordpress_base_url,
            auth_headers,
            category_id,
            description,
            media_id,
            category_thumbnail_meta_key,
        )
        logger(f"[INFO] REST category update succeeded for '{clean_name}'.")
    except Exception as exc:
        logger(f"[WARN] REST thumbnail assignment failed, falling back to wp-admin automation: {exc}")
        set_category_thumbnail_via_wp_admin(
            wordpress_base_url=wordpress_base_url,
            wp_admin_username=wp_admin_username,
            wp_admin_password=wp_admin_password,
            category_id=category_id,
            description=description,
            media_id=media_id,
            logger=logger,
        )
    logger(f"[OK] Updated category {category_id} with media ID {media_id}")


def run_batch(config: BatchConfig, logger: LogCallback = default_logger) -> List[Dict[str, str]]:
    auth_headers = build_wp_auth_header(config.wordpress_username, config.wordpress_app_password)
    api_keys = normalize_gemini_api_keys(config.gemini_api_key, config.gemini_api_keys)
    clients = [genai.Client(api_key=api_key) for api_key in api_keys]
    entries = read_manga_entries(config.mangas_file)
    return run_batch_entries(
        gemini_clients=clients,
        wordpress_base_url=config.wordpress_base_url,
        auth_headers=auth_headers,
        entries=entries,
        prompt_templates=config.prompt_templates,
        wp_admin_username=config.wp_admin_username,
        wp_admin_password=config.wp_admin_password,
        category_thumbnail_meta_key=config.category_thumbnail_meta_key,
        sleep_seconds=config.sleep_seconds,
        logger=logger,
    )


def run_batch_entries(
    gemini_clients: List[genai.Client],
    wordpress_base_url: str,
    auth_headers: Dict[str, str],
    entries: List[Tuple[int, str, str]],
    prompt_templates: Optional[List[str]] = None,
    wp_admin_username: str = "",
    wp_admin_password: str = "",
    category_thumbnail_meta_key: str = "category_thumbnail_id",
    sleep_seconds: int = RATE_LIMIT_SLEEP_SECONDS,
    logger: LogCallback = default_logger,
) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []

    prompt_count = max(len([item for item in (prompt_templates or []) if item.strip()]), 1)

    total_entries = len(entries)

    for index, (category_id, clean_name, language) in enumerate(entries):
        try:
            prompt_template = choose_prompt_template(prompt_templates, index)
            logger(f"[INFO] Processing item {index + 1}/{total_entries}.")
            logger(f"[INFO] Using prompt template #{(index % prompt_count) + 1} for {clean_name}.")
            process_manga(
                gemini_clients=gemini_clients,
                wordpress_base_url=wordpress_base_url,
                auth_headers=auth_headers,
                category_id=category_id,
                clean_name=clean_name,
                language=language,
                prompt_template=prompt_template,
                manga_index=index,
                wp_admin_username=wp_admin_username,
                wp_admin_password=wp_admin_password,
                category_thumbnail_meta_key=category_thumbnail_meta_key,
                logger=logger,
            )
            results.append(
                {
                    "category_id": str(category_id),
                    "title": clean_name,
                    "language": language,
                    "status": "success",
                }
            )
        except Exception as exc:
            logger(f"[ERROR] Failed to process category {category_id} ({clean_name}): {exc}")
            results.append(
                {
                    "category_id": str(category_id),
                    "title": clean_name,
                    "language": language,
                    "status": "failed",
                    "error": str(exc),
                }
            )
        finally:
            if index < total_entries - 1:
                logger(f"[INFO] Sleeping {sleep_seconds} seconds before the next manga.")
                time.sleep(sleep_seconds)

    return results


def main() -> None:
    # Use a WordPress Application Password instead of your main login password.
    # Make sure your theme/plugin exposes 'category_thumbnail_id' to the REST API
    # with register_term_meta so the meta update is accepted.
    wordpress_base_url = load_env("WORDPRESS_BASE_URL")
    wordpress_username = load_env("WORDPRESS_USERNAME")
    wordpress_application_password = load_env("WORDPRESS_APP_PASSWORD")
    gemini_api_key = load_env("GEMINI_API_KEY")

    run_batch(
        BatchConfig(
            wordpress_base_url=wordpress_base_url,
            wordpress_username=wordpress_username,
            wordpress_app_password=wordpress_application_password,
            gemini_api_key=gemini_api_key,
            gemini_api_keys=[],
            wp_admin_username=load_env("WP_ADMIN_USERNAME", ""),
            wp_admin_password=load_env("WP_ADMIN_PASSWORD", ""),
            category_thumbnail_meta_key="category_thumbnail_id",
            prompt_templates=[get_default_prompt_template()],
            sleep_seconds=RATE_LIMIT_SLEEP_SECONDS,
            mangas_file=MANGAS_FILE,
        )
    )


if __name__ == "__main__":
    main()
