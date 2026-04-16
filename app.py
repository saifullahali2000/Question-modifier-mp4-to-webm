import io
import re
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import boto3
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from imageio_ffmpeg import get_ffmpeg_exe
from selenium import webdriver
from selenium.common.exceptions import ElementClickInterceptedException, StaleElementReferenceException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


VIDEO_URL_PATTERN = re.compile(
    r'https?:\\?/\\?/[^"\s<>]+?\.(?:mp4|mov|webm)(?:\?[^"\s<>]*)?',
    flags=re.IGNORECASE,
)
# Match full <source ...> through > so attributes after src (e.g. type="video/mp4") are rewritten to webm.
SOURCE_TAG_PATTERN = re.compile(
    r'(<source[^>]*\bsrc\s*=\s*["\'])([^"\']+)(["\'][^>]*>)',
    flags=re.IGNORECASE,
)
TYPE_ATTR_PATTERN = re.compile(r'\btype\s*=\s*["\'][^"\']*["\']', flags=re.IGNORECASE)


def extract_video_url(question_text: str) -> str:
    if pd.isna(question_text):
        return ""

    text = unescape(str(question_text))
    source_match = SOURCE_TAG_PATTERN.search(text)
    if source_match:
        candidate = source_match.group(2).replace("\\/", "/")
        match = VIDEO_URL_PATTERN.search(candidate)
        if match:
            return match.group(0).replace("\\/", "/")

    match = VIDEO_URL_PATTERN.search(text)
    return match.group(0).replace("\\/", "/") if match else ""


def update_question_text_with_webm(question_text: str, webm_url: str) -> str:
    if pd.isna(question_text):
        return ""

    text = str(question_text)
    source_match = SOURCE_TAG_PATTERN.search(text)
    if not source_match:
        return text

    full_source = source_match.group(0)
    prefix, _, suffix = source_match.groups()
    new_source = f"{prefix}{webm_url}{suffix}"

    if TYPE_ATTR_PATTERN.search(new_source):
        new_source = TYPE_ATTR_PATTERN.sub('type="video/webm"', new_source, count=1)
    else:
        new_source = new_source.replace(">", ' type="video/webm">', 1)

    return text.replace(full_source, new_source, 1)


def to_s3_key(folder: str, question_id: str, source_url: str) -> str:
    parsed = urlparse(source_url)
    source_name = Path(parsed.path).name
    base_name = Path(source_name).stem if source_name else str(question_id).strip()

    safe_base = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(base_name)).strip("_")
    if not safe_base:
        safe_base = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(question_id)).strip("_") or "video"

    normalized_folder = folder.strip().strip("/")
    file_name = f"{safe_base}.webm"
    return f"{normalized_folder}/{file_name}" if normalized_folder else file_name


def download_file(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def convert_to_webm(input_path: Path, output_path: Path) -> None:
    ffmpeg_path = get_ffmpeg_exe()
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libvpx-vp9",
        "-deadline",
        "realtime",
        "-cpu-used",
        "8",
        "-b:v",
        "0",
        "-crf",
        "38",
        "-c:a",
        "libopus",
        "-threads",
        "4",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def build_s3_client(access_key: str, secret_key: str, region: str):
    return boto3.client(
        "s3",
        aws_access_key_id=access_key.strip(),
        aws_secret_access_key=secret_key.strip(),
        region_name=region.strip(),
    )


def upload_webm_to_s3(s3_client, bucket: str, key: str, file_path: Path) -> str:
    s3_client.upload_file(
        str(file_path),
        bucket.strip(),
        key,
        ExtraArgs={
            "ContentType": "video/webm",
            "ACL": "public-read",
        },
    )
    region = s3_client.meta.region_name or "us-east-1"
    if region == "us-east-1":
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Topin Prod Question ID", "Question Text"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_cols}")

    output = df[["Topin Prod Question ID", "Question Text"]].copy()
    output["Extracted Source URL"] = output["Question Text"].apply(extract_video_url)
    output["Uploaded WebM URL"] = ""
    output["Updated Question Text"] = output["Question Text"]
    output["Status"] = "Pending"
    return output


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def build_final_output_csv(result_df: pd.DataFrame) -> pd.DataFrame:
    final_df = result_df[["Topin Prod Question ID", "Updated Question Text"]].copy()
    final_df = final_df.rename(columns={"Updated Question Text": "Question Text"})
    return final_df


def process_single_row(row: pd.Series, s3_client, s3_bucket: str, s3_folder: str) -> dict:
    question_id = row["Topin Prod Question ID"]
    source_url = row["Extracted Source URL"]
    question_text = row["Question Text"]

    if not source_url:
        return {
            "uploaded_webm_url": "",
            "updated_question_text": question_text,
            "status": "Skipped: URL not found",
        }

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_name = Path(urlparse(source_url).path).name or f"{question_id}.bin"
            input_path = temp_path / input_name
            output_path = temp_path / f"{Path(input_name).stem}.webm"

            download_file(source_url, input_path)

            # If input is already webm, skip conversion and re-upload directly.
            if Path(input_name).suffix.lower() == ".webm":
                upload_source = input_path
            else:
                convert_to_webm(input_path, output_path)
                upload_source = output_path

            s3_key = to_s3_key(s3_folder, str(question_id), source_url)
            webm_url = upload_webm_to_s3(s3_client, s3_bucket, s3_key, upload_source)

            return {
                "uploaded_webm_url": webm_url,
                "updated_question_text": update_question_text_with_webm(question_text, webm_url),
                "status": "Success",
            }
    except Exception as row_exc:
        return {
            "uploaded_webm_url": "",
            "updated_question_text": question_text,
            "status": f"Failed: {format_exception_for_status(row_exc)}",
        }


def get_csrf_token_from_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    token_input = soup.find("input", attrs={"name": "csrfmiddlewaretoken"})
    return token_input.get("value", "") if token_input else ""


def get_admin_base_url(admin_url: str) -> str:
    parsed = urlparse(admin_url)
    path = parsed.path or "/admin/"
    admin_index = path.find("/admin/")
    if admin_index == -1:
        raise ValueError("Provided URL is not a Django admin URL (missing '/admin/').")
    admin_path = path[: admin_index + len("/admin/")]
    return f"{parsed.scheme}://{parsed.netloc}{admin_path}"


def get_relative_path(full_url: str) -> str:
    parsed = urlparse(full_url)
    path = parsed.path or "/"
    return f"{path}?{parsed.query}" if parsed.query else path


def login_django_admin(session: requests.Session, admin_url: str, username: str, password: str) -> None:
    admin_base_url = get_admin_base_url(admin_url)
    login_url = admin_base_url.rstrip("/") + "/login/"
    login_page = session.get(login_url, timeout=30)
    login_page.raise_for_status()

    soup = BeautifulSoup(login_page.text, "html.parser")
    login_form = soup.find("form")
    if login_form is None:
        raise ValueError("Could not find login form on Django login page.")

    action = login_form.get("action", "").strip()
    if action.startswith("http://") or action.startswith("https://"):
        submit_url = action
    elif action.startswith("/"):
        base_admin_root = admin_base_url.split("/admin/")[0]
        submit_url = f"{base_admin_root}{action}"
    else:
        submit_url = login_url

    payload = {}
    for input_tag in login_form.select("input[name]"):
        name = input_tag.get("name")
        if not name:
            continue
        payload[name] = input_tag.get("value", "")

    csrf_token = payload.get("csrfmiddlewaretoken") or get_csrf_token_from_html(login_page.text)
    if csrf_token:
        payload["csrfmiddlewaretoken"] = csrf_token

    payload["username"] = username
    payload["password"] = password
    payload["next"] = get_relative_path(admin_url)

    headers = {"Referer": login_url}
    response = session.post(submit_url, data=payload, headers=headers, timeout=30, allow_redirects=True)
    response.raise_for_status()
    if "/login/" in response.url:
        response_soup = BeautifulSoup(response.text, "html.parser")
        error_list = [err.get_text(strip=True) for err in response_soup.select(".errornote, .errorlist li")]
        if not error_list:
            error_list = ["Invalid credentials or user does not have admin/staff access."]
        raise ValueError("Django login failed: " + " | ".join(error_list))


def find_question_change_url(session: requests.Session, admin_url: str, topin_prod_question_id: str) -> str:
    list_url = admin_url.rstrip("/") + "/"
    params = {"q": str(topin_prod_question_id)}
    response = session.get(list_url, params=params, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    result_table = soup.find("table", id="result_list")
    if result_table is None:
        raise ValueError("Could not find search result table in Django admin.")

    matched_link = None
    target_id = str(topin_prod_question_id).strip()
    for row in result_table.select("tbody tr"):
        id_link = row.select_one("th a")
        if id_link is None:
            continue
        row_id = id_link.get_text(strip=True)
        if row_id == target_id:
            matched_link = id_link
            break

    if matched_link is None:
        # Fallback to first row if exact text match not found.
        matched_link = result_table.select_one("tbody tr th a")
    if matched_link is None:
        raise ValueError(f"No question found for Topin Prod Question ID: {topin_prod_question_id}")

    href = matched_link.get("href", "")
    if not href:
        raise ValueError("Found question row but change link is missing.")

    if href.startswith("http://") or href.startswith("https://"):
        return href

    base_admin_root = admin_url.split("/admin/")[0]
    if href.startswith("/"):
        return f"{base_admin_root}{href}"
    return f"{admin_url.rstrip('/')}/{href.lstrip('./')}"


def update_question_content(
    session: requests.Session, change_url: str, updated_question_text: str, topin_prod_question_id: str
) -> str:
    form_page = session.get(change_url, timeout=30)
    form_page.raise_for_status()

    soup = BeautifulSoup(form_page.text, "html.parser")
    form = soup.find("form", id="question_form") or soup.find("form", id="nkb_question_form") or soup.find("form")
    if form is None:
        raise ValueError("Could not find edit form on question change page.")

    csrf_token = get_csrf_token_from_html(form_page.text)
    if not csrf_token:
        raise ValueError("Could not find CSRF token on change page.")

    payload = {}
    for input_tag in form.select("input[name]"):
        input_type = (input_tag.get("type") or "").lower()
        name = input_tag.get("name")
        if not name:
            continue
        if input_type in {"submit", "file"}:
            continue
        if input_type in {"checkbox", "radio"}:
            if input_tag.has_attr("checked"):
                payload[name] = input_tag.get("value", "on")
            continue
        payload[name] = input_tag.get("value", "")

    textarea_names = set()
    textarea_by_id = {}
    for textarea_tag in form.select("textarea[name]"):
        name = textarea_tag.get("name")
        if name:
            textarea_names.add(name)
            payload[name] = textarea_tag.text or ""
            tag_id = textarea_tag.get("id")
            if tag_id:
                textarea_by_id[tag_id] = name

    for select_tag in form.select("select[name]"):
        name = select_tag.get("name")
        if not name:
            continue
        selected = select_tag.select_one("option[selected]")
        if selected is not None:
            payload[name] = selected.get("value", "")
        else:
            first_option = select_tag.select_one("option")
            payload[name] = first_option.get("value", "") if first_option else ""

    payload["csrfmiddlewaretoken"] = csrf_token
    if "_save" not in payload:
        payload["_save"] = "Save"

    # Prefer exact mapping by visible field labels from Django form rows.
    target_names = set()
    for label in form.select("label[for]"):
        label_text = label.get_text(" ", strip=True).lower()
        field_id = label.get("for", "")
        if not field_id:
            continue
        mapped_name = textarea_by_id.get(field_id)
        if not mapped_name:
            continue

        if label_text == "content:" or label_text == "content":
            target_names.add(mapped_name)
        if label_text in {"content [en]:", "content [en]", "content[en]:", "content[en]"}:
            target_names.add(mapped_name)

    # Fallback to name-based matching only for content + content[en].
    if not target_names:
        for key in textarea_names:
            key_lower = key.lower()
            normalized = key_lower.replace("_", "").replace("-", "")
            is_base_content = normalized == "content"
            is_english_content = (
                key_lower == "content[en]"
                or normalized == "contenten"
                or key_lower.startswith("content[en]")
            )
            if is_base_content or is_english_content:
                target_names.add(key)

    content_updated = False
    for key in target_names:
        payload[key] = updated_question_text
        content_updated = True

    if not content_updated:
        raise ValueError(
            f"Could not find target fields (content/content[en]) for Topin Prod Question ID: {topin_prod_question_id}"
        )

    headers = {"Referer": change_url}
    save_response = session.post(change_url, data=payload, headers=headers, timeout=30, allow_redirects=True)
    save_response.raise_for_status()
    if "Please correct the errors below" in save_response.text:
        error_soup = BeautifulSoup(save_response.text, "html.parser")
        errors = [item.get_text(" ", strip=True) for item in error_soup.select(".errornote, .errorlist li")]
        # Field specific messages are often shown near the labels.
        field_errors = []
        for field_row in error_soup.select(".form-row.errors, .fieldBox.errors"):
            label = field_row.select_one("label")
            err = field_row.select_one(".errorlist")
            if label and err:
                field_errors.append(f"{label.get_text(' ', strip=True)} -> {err.get_text(' ', strip=True)}")

        combined = errors + field_errors
        if not combined:
            combined = ["Unknown form validation error from Django admin."]

        raise ValueError("Django rejected update: " + " | ".join(combined[:8]))

    # Verify save actually happened and data persisted.
    # Django/admin themes can sanitize HTML/whitespace, so exact-string compares are too strict.
    low_response = save_response.text.lower()
    success_markers = ["was changed successfully", "changed successfully", "successfully updated"]
    success_message_present = any(marker in low_response for marker in success_markers)
    if not success_message_present:
        response_soup = BeautifulSoup(save_response.text, "html.parser")
        success_message_present = bool(response_soup.select(".messagelist .success, .alert-success"))

    def normalize_for_compare(text: str) -> str:
        value = unescape(str(text or ""))
        value = re.sub(r"\s+", " ", value).strip()
        value = value.replace('\\"', '"').replace("\\/", "/")
        return value

    if not success_message_present:
        # Some admin themes still redirect without standard message; verify by re-fetching page.
        verify_page = session.get(change_url, timeout=30)
        verify_page.raise_for_status()
        verify_soup = BeautifulSoup(verify_page.text, "html.parser")
        persisted = False
        expected = normalize_for_compare(updated_question_text)
        for key in target_names:
            textarea = verify_soup.select_one(f'textarea[name="{key}"]')
            if not textarea:
                continue
            actual = normalize_for_compare(textarea.text or "")
            # Accept exact, contains, or reverse-contains to handle admin formatting/sanitization.
            if actual == expected or expected in actual or actual in expected:
                persisted = True
                break
        if not persisted:
            raise ValueError(
                "Save request completed but could not verify persistence in content fields "
                "(admin may sanitize markup; please spot-check one saved row)."
            )

    return "Success"


def build_visible_chrome_driver() -> webdriver.Chrome:
    options = Options()
    options.add_experimental_option("detach", True)
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def normalize_topin_prod_question_id(value) -> str:
    """Avoid empty / 'nan' / Excel float IDs in the first CSV rows (BOM, NaN, 1.0-style ints)."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    text = str(value).strip()
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff").strip()
    if text.lower() in {"nan", "none", "nat", "<na>"}:
        return ""
    return text


def format_exception_for_status(exc: BaseException) -> str:
    """Selenium/ChromeDriver sometimes leaves str(exc) empty; include type and traceback hint."""
    parts = [type(exc).__name__]
    msg = (getattr(exc, "msg", None) or str(exc) or "").strip()
    if msg:
        parts.append(msg)
    if getattr(exc, "args", None):
        extra = " | ".join(repr(a) for a in exc.args if a and str(a).strip() != msg)
        if extra:
            parts.append(extra)
    if len(parts) == 1:
        parts.append(traceback.format_exception_only(type(exc), exc)[-1].strip())
    return ": ".join(parts) if len(parts) > 1 else parts[0]


def _fill_changelist_search(driver: webdriver.Chrome, wait: WebDriverWait, topin_id: str):
    """Focus the changelist search field, clear it, type the id. Returns that input for paired form submit."""
    if not (topin_id or "").strip():
        raise ValueError("Topin Prod Question ID is empty for this CSV row (check the first data row).")

    wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
    wait.until(
        EC.presence_of_element_located(
            (By.CSS_SELECTOR, "#changelist-search, #changelist, form#changelist-search")
        )
    )

    selectors = (
        (By.CSS_SELECTOR, "#changelist-search input[name='q']"),
        (By.CSS_SELECTOR, "form#changelist-search input[name='q']"),
        (By.CSS_SELECTOR, "input#searchbar[name='q']"),
        (By.NAME, "q"),
    )
    search_box = None
    for by, sel in selectors:
        try:
            el = WebDriverWait(driver, min(15, getattr(wait, "_timeout", 30) or 30)).until(
                EC.element_to_be_clickable((by, sel))
            )
            if el.is_displayed():
                search_box = el
                break
        except (TimeoutException, ElementClickInterceptedException):
            continue
    if search_box is None:
        raise ValueError("Could not find a clickable changelist search field (name=q).")

    _safe_click(driver, search_box)
    search_box.send_keys(Keys.CONTROL + "a")
    search_box.send_keys(Keys.BACKSPACE)
    search_box.send_keys(topin_id)

    entered = (search_box.get_attribute("value") or "").strip()
    if entered != topin_id.strip():
        driver.execute_script(
            "arguments[0].value = arguments[1];"
            "arguments[0].dispatchEvent(new Event('input', {bubbles: true}));"
            "arguments[0].dispatchEvent(new Event('change', {bubbles: true}));",
            search_box,
            topin_id,
        )

    return search_box


def _safe_click(driver: webdriver.Chrome, element) -> None:
    """Normal click with scroll; JS click if a sticky header/footer intercepts the hit target."""
    driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", element)
    try:
        element.click()
    except ElementClickInterceptedException:
        driver.execute_script("arguments[0].click();", element)


def _click_django_changelist_search(driver, wait: WebDriverWait, search_input) -> None:
    """Submit search using the same <form> as the filled ``search_input`` (avoids a hidden name=q + wrong submit on row 1)."""
    form = search_input.find_element(By.XPATH, "./ancestor::form[1]")
    for el in form.find_elements(By.CSS_SELECTOR, "input[type='submit'], button[type='submit']"):
        if el.is_displayed() and el.is_enabled():
            _safe_click(driver, el)
            return
    try:
        search_input.send_keys(Keys.RETURN)
    except Exception:
        driver.execute_script("arguments[0].submit();", form)


def _changelist_search_applied(driver, topin_id: str) -> bool:
    """True once the list URL reflects an admin search for this id (GET ?q=...)."""
    raw = parse_qs(urlparse(driver.current_url).query).get("q", [""])
    if not raw:
        return False
    return unquote(raw[0]).strip() == str(topin_id).strip()


def _result_list_link_for_topin_id(driver, topin_id: str):
    """Return the changelist link for this Topin ID (visible text or href), or False while loading."""
    target = str(topin_id).strip()
    try:
        table = driver.find_element(By.ID, "result_list")
    except Exception:
        return False
    for row in table.find_elements(By.CSS_SELECTOR, "tbody tr"):
        for link in row.find_elements(
            By.CSS_SELECTOR,
            "th a, td.field-topin_prod_question_id a, td.field-id a, td a",
        ):
            try:
                href = (link.get_attribute("href") or "").strip()
                if link.text.strip() == target or target in href:
                    return link
            except Exception:
                continue
    return False


def _wait_django_save_success(driver, wait: WebDriverWait) -> None:
    """Django admin themes vary; match common success copy and classes."""

    def save_done(drv):
        low = drv.page_source.lower()
        if "please correct the errors below" in low:
            raise ValueError("Django admin reported validation errors after save; check required fields in Chrome.")
        if (
            "was changed successfully" in low
            or "was added successfully" in low
            or "successfully updated" in low
        ):
            return True
        if drv.find_elements(By.CSS_SELECTOR, ".messagelist li.success, .messagelist .success"):
            return True
        if drv.find_elements(By.CSS_SELECTOR, ".alert-success"):
            return True
        return False

    wait.until(save_done)


def update_question_with_selenium(
    driver: webdriver.Chrome, admin_url: str, topin_id: str, updated_text: str, wait_seconds: int = 45
) -> str:
    wait = WebDriverWait(driver, wait_seconds)
    driver.get(admin_url)

    search_field = _fill_changelist_search(driver, wait, str(topin_id).strip())
    _click_django_changelist_search(driver, wait, search_field)

    wait.until(EC.presence_of_element_located((By.ID, "result_list")))
    try:
        WebDriverWait(driver, min(15, wait_seconds)).until(lambda d: _changelist_search_applied(d, topin_id))
    except TimeoutException:
        pass

    def id_link_ready(drv):
        el = _result_list_link_for_topin_id(drv, topin_id)
        if el is False:
            return False
        try:
            return el if el.is_displayed() and el.is_enabled() else False
        except Exception:
            return False

    try:
        link = wait.until(id_link_ready)
    except TimeoutException as exc:
        raise ValueError(
            f"No #result_list link for Topin ID {topin_id!r} after search within {wait_seconds}s "
            "(check filters, ID spelling, or that this question exists in this admin list)."
        ) from exc

    for attempt in range(4):
        try:
            _safe_click(driver, link)
            break
        except StaleElementReferenceException:
            if attempt == 3:
                raise ValueError(
                    f"Stale DOM when opening edit page for Topin ID {topin_id!r}; try again."
                ) from None
            link = wait.until(id_link_ready)

    wait.until(
        EC.presence_of_element_located(
            (
                By.CSS_SELECTOR,
                "textarea[name='content'], textarea[name='content[en]'], textarea[name^='content']",
            )
        )
    )

    content_candidates = [
        "content",
        "content[en]",
        "content_en",
        "content-en",
        "contenten",
    ]
    updated_count = 0
    for field_name in content_candidates:
        elems = driver.find_elements(By.NAME, field_name)
        for elem in elems:
            elem.clear()
            elem.send_keys(updated_text)
            updated_count += 1

    if updated_count == 0:
        return "Failed: content/content[en] fields not found on page"

    save_button = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "input[name='_save'], button[name='_save']"))
    )
    _safe_click(driver, save_button)

    try:
        _wait_django_save_success(driver, wait)
    except TimeoutException as exc:
        raise ValueError(
            f"No Django success message within {wait_seconds}s after save for {topin_id!r} "
            "(save may still have worked — check the question in admin)."
        ) from exc
    return "Success"


st.set_page_config(page_title="Topin Video URL Processor", page_icon="🎬", layout="centered")
st.title("Topin Video URL Processor")
st.write(
    "Upload CSV with `Topin Prod Question ID` and `Question Text`, then convert and upload videos to S3 as WebM."
)

with st.expander("AWS S3 Configuration", expanded=True):
    aws_access_key = st.text_input("AWS Access Key ID", type="password")
    aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
    aws_region = st.text_input("AWS Region", value="ap-south-1")
    s3_bucket = st.text_input("S3 Bucket Name")
    s3_folder = st.text_input("S3 Folder", value="placement-happenings/webcoding-gif")
    max_workers = st.slider("Parallel workers", min_value=1, max_value=8, value=4)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        result_df = process_dataframe(input_df)

        total_rows = len(result_df)
        extracted_rows = int((result_df["Extracted Source URL"] != "").sum())
        missing_rows = total_rows - extracted_rows

        st.write(f"Total rows: **{total_rows}**")
        st.write(f"Rows with extracted URL: **{extracted_rows}**")
        st.write(f"Rows with no URL found: **{missing_rows}**")

        if st.button("Start Convert + Upload to S3", type="primary"):
            if not all([aws_access_key, aws_secret_key, aws_region, s3_bucket]):
                st.error("Fill AWS Access Key, Secret Key, Region, and Bucket before processing.")
                st.stop()

            s3_client = build_s3_client(aws_access_key, aws_secret_key, aws_region)
            progress = st.progress(0)
            status_text = st.empty()
            success_count = 0

            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for idx, row in result_df.iterrows():
                    future = executor.submit(process_single_row, row, s3_client, s3_bucket, s3_folder)
                    futures[future] = idx

                completed = 0
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    result_df.at[idx, "Uploaded WebM URL"] = result["uploaded_webm_url"]
                    result_df.at[idx, "Updated Question Text"] = result["updated_question_text"]
                    result_df.at[idx, "Status"] = result["status"]
                    if result["status"] == "Success":
                        success_count += 1

                    completed += 1
                    question_id = result_df.at[idx, "Topin Prod Question ID"]
                    status_text.write(f"Processed `{question_id}` ({completed}/{total_rows})")
                    progress.progress(completed / total_rows)

            status_text.write("Done.")
            st.success(f"Finished. Successfully uploaded: {success_count}/{total_rows}")

        st.subheader("Preview")
        st.dataframe(
            result_df[
                [
                    "Topin Prod Question ID",
                    "Extracted Source URL",
                    "Uploaded WebM URL",
                    "Status",
                ]
            ].head(20),
            use_container_width=True,
        )

        if missing_rows > 0:
            unresolved_ids = result_df.loc[
                result_df["Extracted Source URL"] == "", "Topin Prod Question ID"
            ].head(20)
            st.warning("Some rows do not include a supported media URL (.mp4/.mov/.webm).")
            st.dataframe(unresolved_ids.to_frame(name="Topin Prod Question ID"), use_container_width=True)

        final_output_df = build_final_output_csv(result_df)
        st.download_button(
            label="Download Output CSV",
            data=to_csv_bytes(final_output_df),
            file_name="topin_question_webm_output.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Could not process file: {exc}")


st.divider()
st.subheader("Step 2: Update Django Admin from CSV")
st.write(
    "Upload the final CSV (`Topin Prod Question ID`, `Question Text`) and update matching questions in Django admin."
)

django_admin_url = st.text_input(
    "Django Question Admin URL",
    value="https://nxtwave-assessments-backend-topin-prod-apis.ccbp.in/admin/nkb_question/question/",
)
django_username = st.text_input("Django Username", type="password")
django_password = st.text_input("Django Password", type="password")
use_visible_chrome = st.checkbox("Use visible Chrome WebDriver (live debugging)", value=False)
django_csv_file = st.file_uploader("Upload final CSV for Django update", type=["csv"], key="django_csv_uploader")

if django_csv_file is not None:
    try:
        django_df = pd.read_csv(django_csv_file, encoding="utf-8-sig")
        required_django_cols = {"Topin Prod Question ID", "Question Text"}
        missing_cols = required_django_cols.difference(django_df.columns)
        if missing_cols:
            st.error(f"CSV missing columns: {', '.join(sorted(missing_cols))}")
        else:
            st.write(f"Rows ready for Django update: **{len(django_df)}**")
            if st.button("Start Django Update", type="primary"):
                if not all([django_admin_url, django_username, django_password]):
                    st.error("Please provide Django admin URL, username, and password.")
                    st.stop()

                progress = st.progress(0)
                status_box = st.empty()
                results = []
                total = len(django_df)
                success_count = 0

                if use_visible_chrome:
                    driver = None
                    try:
                        driver = build_visible_chrome_driver()
                        driver.get(get_admin_base_url(django_admin_url).rstrip("/") + "/login/")
                        WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.NAME, "username"))).send_keys(
                            django_username
                        )
                        driver.find_element(By.NAME, "password").send_keys(django_password)
                        driver.find_element(By.CSS_SELECTOR, "input[type='submit'],button[type='submit']").click()

                        if "/login/" in driver.current_url:
                            raise ValueError("Login failed in Chrome. Check credentials or admin access.")

                        # Load changelist once so the first CSV row is not racing post-login redirects / duplicate name=q.
                        driver.get(django_admin_url)
                        WebDriverWait(driver, 30).until(
                            EC.presence_of_element_located(
                                (By.CSS_SELECTOR, "#changelist-search, form#changelist-search, #changelist")
                            )
                        )

                        for n_done, (_, row) in enumerate(django_df.iterrows(), start=1):
                            topin_id = normalize_topin_prod_question_id(row["Topin Prod Question ID"])
                            updated_text = str(row["Question Text"])
                            if pd.isna(row["Question Text"]):
                                updated_text = ""
                            status_box.write(f"[Chrome] Updating `{topin_id or '(missing id)'}` ...")
                            try:
                                if not topin_id:
                                    raise ValueError(
                                        "Topin Prod Question ID is missing in this row "
                                        "(common on row 1: UTF-8 BOM, blank cell, or Excel number format)."
                                    )
                                status = update_question_with_selenium(driver, django_admin_url, topin_id, updated_text)
                                if status == "Success":
                                    success_count += 1
                            except Exception as row_exc:
                                status = f"Failed: {format_exception_for_status(row_exc)}"

                            results.append(
                                {
                                    "Topin Prod Question ID": topin_id,
                                    "Django Update Status": status,
                                }
                            )
                            progress.progress(n_done / total)
                    except Exception as chrome_exc:
                        st.error(f"Chrome WebDriver flow failed: {chrome_exc}")
                    finally:
                        if driver is not None:
                            driver.quit()
                else:
                    session = requests.Session()
                    session.headers.update({"User-Agent": "TopinVideoProcessor/1.0"})

                    try:
                        login_django_admin(session, django_admin_url, django_username, django_password)
                    except Exception as login_exc:
                        st.error(f"Django login failed: {login_exc}")
                        st.stop()

                    for n_done, (_, row) in enumerate(django_df.iterrows(), start=1):
                        topin_id = normalize_topin_prod_question_id(row["Topin Prod Question ID"])
                        updated_text = str(row["Question Text"])
                        if pd.isna(row["Question Text"]):
                            updated_text = ""
                        status_box.write(f"Updating `{topin_id or '(missing id)'}` in Django ...")
                        try:
                            if not topin_id:
                                raise ValueError(
                                    "Topin Prod Question ID is missing in this row "
                                    "(check UTF-8 BOM / blank first data cell / Excel formatting)."
                                )
                            change_url = find_question_change_url(session, django_admin_url, topin_id)
                            status = update_question_content(session, change_url, updated_text, topin_id)
                            success_count += 1
                        except Exception as row_exc:
                            status = f"Failed: {format_exception_for_status(row_exc)}"

                        results.append(
                            {
                                "Topin Prod Question ID": topin_id,
                                "Django Update Status": status,
                            }
                        )
                        progress.progress(n_done / total)

                status_box.write("Django update run complete.")
                st.success(f"Updated successfully: {success_count}/{total}")
                result_status_df = pd.DataFrame(results)
                st.dataframe(result_status_df.head(50), use_container_width=True)
                st.download_button(
                    label="Download Django Update Status CSV",
                    data=to_csv_bytes(result_status_df),
                    file_name="django_update_status.csv",
                    mime="text/csv",
                )
    except Exception as exc:
        st.error(f"Could not read Django update CSV: {exc}")
