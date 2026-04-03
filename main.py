from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict
from lib.imageProcessing import (
    getFaceEmbedding,
    cosineSimilarity,
    checkLiveness,
    checkImageQuality,
    resize_base64_image_to_jpeg_bytes,
    resize_base64_image_to_jpeg_base64,
    TARGET_WIDTH,
    TARGET_HEIGHT,
)
from lib.security import _cleanup, get_user_id, verify_hmac_request, issue_hmac_key
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timezone, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import os
import base64
import json
import logging
import time

app = FastAPI()

logger = logging.getLogger("ai-microservice")
logging.basicConfig(level=logging.INFO)

_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = 10  # requests per window
_RATE_STORE: Dict[str, Dict[str, int]] = {}


@app.get("/")
async def root_health():
    return {"status": "ok", "service": "ai-microservice"}


load_dotenv()

url = os.environ.get("SUPABASE_URL") 
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(url, key)


def _env_to_bool(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SAVE_ATTENDANCE_IMAGE = _env_to_bool(os.environ.get("SAVE_ATTENDANCE_IMAGE"), default=True)
SAVE_NORMALIZED_DEBUG_IMAGES = _env_to_bool(os.environ.get("SAVE_NORMALIZED_DEBUG_IMAGES"), default=True)

# Liveness detection configuration
LIVENESS_ENABLED = _env_to_bool(os.environ.get("LIVENESS_DETECTION_ENABLED"), default=True)
LIVENESS_CONFIDENCE_THRESHOLD = float(os.environ.get("LIVENESS_CONFIDENCE_THRESHOLD", "0.7"))
LIVENESS_FAIL_ACTION = os.environ.get("LIVENESS_FAIL_ACTION", "block")  # 'block' or 'warn'
MATCH_SIMILARITY_THRESHOLD = float(os.environ.get("MATCH_SIMILARITY_THRESHOLD", "0.7"))
IMAGE_QUALITY_CHECK_ENABLED = _env_to_bool(os.environ.get("IMAGE_QUALITY_CHECK_ENABLED"), default=True)
MIN_IMAGE_BRIGHTNESS = float(os.environ.get("MIN_IMAGE_BRIGHTNESS", "55"))
MIN_IMAGE_SHARPNESS = float(os.environ.get("MIN_IMAGE_SHARPNESS", "45"))


def _normalize_embedding(value):
    if value is None:
        return None

    if isinstance(value, list):
        try:
            return [float(x) for x in value]
        except Exception:
            return None

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            pass

        try:
            compact = stripped.strip('[]')
            if not compact:
                return None
            return [float(x.strip()) for x in compact.split(',') if x.strip()]
        except Exception:
            return None

    return None


def _extract_embeddings_from_profile_row(profile_row: dict | None) -> list[list[float]]:
    if not profile_row:
        return []

    candidate_embeddings: list[list[float]] = []

    direct_keys = [
        "face_embedding_front",
        "face_embedding_left",
        "face_embedding_right",
        "embedding_front",
        "embedding_left",
        "embedding_right",
    ]

    for key in direct_keys:
        normalized = _normalize_embedding(profile_row.get(key))
        if normalized:
            candidate_embeddings.append(normalized)

    for aggregate_key in ["face_embeddings", "embeddings"]:
        aggregate_value = profile_row.get(aggregate_key)

        if isinstance(aggregate_value, dict):
            for nested_key in ["front", "left", "right", "embedding_front", "embedding_left", "embedding_right"]:
                normalized = _normalize_embedding(aggregate_value.get(nested_key))
                if normalized:
                    candidate_embeddings.append(normalized)

        if isinstance(aggregate_value, list):
            for item in aggregate_value:
                normalized = _normalize_embedding(item)
                if normalized:
                    candidate_embeddings.append(normalized)

    unique_embeddings: list[list[float]] = []
    seen = set()
    for emb in candidate_embeddings:
        signature = tuple(round(float(value), 8) for value in emb)
        if signature in seen:
            continue
        seen.add(signature)
        unique_embeddings.append(emb)

    return unique_embeddings


def _fetch_profile_embeddings(user_id: str) -> list[list[float]]:
    select_variants = [
        "face_embedding_front, face_embedding_left, face_embedding_right",
        "embedding_front, embedding_left, embedding_right",
        "face_embeddings",
        "embeddings",
    ]

    for select_clause in select_variants:
        try:
            profile_result = (
                supabase.table("profiles")
                .select(select_clause)
                .eq("id", user_id)
                .single()
                .execute()
            )

            row = profile_result.data if profile_result and profile_result.data else None
            embeddings = _extract_embeddings_from_profile_row(row)
            if embeddings:
                logger.info("Loaded %d profile embeddings user=%s via select=%s", len(embeddings), user_id, select_clause)
                return embeddings
        except Exception as err:
            logger.warning(
                "Profile embedding fetch variant failed user=%s select=%s err=%s",
                user_id,
                select_clause,
                str(err),
            )

    return []


def _save_profile_embeddings(user_id: str, front_embedding: list[float], left_embedding: list[float], right_embedding: list[float]) -> bool:
    payload_variants = [
        {
            "face_embedding_front": front_embedding,
            "face_embedding_left": left_embedding,
            "face_embedding_right": right_embedding,
        },
        {
            "embedding_front": front_embedding,
            "embedding_left": left_embedding,
            "embedding_right": right_embedding,
        },
        {
            "face_embeddings": {
                "front": front_embedding,
                "left": left_embedding,
                "right": right_embedding,
            },
        },
        {
            "embeddings": [front_embedding, left_embedding, right_embedding],
        },
    ]

    for payload in payload_variants:
        try:
            (
                supabase.table("profiles")
                .update(payload)
                .eq("id", user_id)
                .execute()
            )
            logger.info("Saved profile embeddings user=%s using payload keys=%s", user_id, list(payload.keys()))
            return True
        except Exception as err:
            logger.warning(
                "Profile embedding save variant failed user=%s keys=%s err=%s",
                user_id,
                list(payload.keys()),
                str(err),
            )

    return False


def _save_attendance_image_jpg(image_bytes: bytes, user_id: str, class_id: str) -> str:

    output_dir = os.path.join(os.getcwd(), 'captures', 'attendance')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time() * 1000)
    safe_user_id = str(user_id).replace('/', '_').replace('\\', '_')
    safe_class_id = str(class_id).replace('/', '_').replace('\\', '_')
    file_name = f"{safe_user_id}_{safe_class_id}_{timestamp}.jpg"
    file_path = os.path.join(output_dir, file_name)

    with open(file_path, 'wb') as image_file:
        image_file.write(image_bytes)

    return file_path


def _save_normalized_debug_image(image_bytes: bytes, category: str, user_id: str, label: str, class_id: str | None = None) -> str:
    output_dir = os.path.join(os.getcwd(), 'captures', 'normalized', category)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = int(time.time() * 1000)
    safe_user_id = str(user_id).replace('/', '_').replace('\\', '_')
    safe_label = str(label).replace('/', '_').replace('\\', '_')
    file_name = f"{safe_user_id}_{safe_label}_{timestamp}_{TARGET_WIDTH}x{TARGET_HEIGHT}.jpg"

    if class_id:
        safe_class_id = str(class_id).replace('/', '_').replace('\\', '_')
        file_name = f"{safe_class_id}_{file_name}"

    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'wb') as image_file:
        image_file.write(image_bytes)

    return file_path


def _save_register_normalized_images(
    front_bytes: bytes,
    left_bytes: bytes,
    right_bytes: bytes,
    user_id: str,
    capture_source: str,
) -> dict[str, str]:
    if not SAVE_NORMALIZED_DEBUG_IMAGES:
        return {}

    category = f"embedding/{capture_source}"

    return {
        "front": _save_normalized_debug_image(front_bytes, category, user_id, "front"),
        "left": _save_normalized_debug_image(left_bytes, category, user_id, "left"),
        "right": _save_normalized_debug_image(right_bytes, category, user_id, "right"),
    }


def _save_attendance_normalized_image(image_bytes: bytes, user_id: str, class_id: str) -> str | None:
    if not SAVE_NORMALIZED_DEBUG_IMAGES:
        return None

    return _save_normalized_debug_image(image_bytes, "attendance", user_id, "capture", class_id)


def _upload_attendance_image_to_storage(image_bytes: bytes, user_id: str, class_id: str) -> str | None:
    timestamp = int(time.time() * 1000)
    safe_user_id = str(user_id).replace('/', '_').replace('\\', '_')
    safe_class_id = str(class_id).replace('/', '_').replace('\\', '_')
    storage_path = f"{safe_class_id}/{safe_user_id}_{timestamp}.jpg"

    supabase.storage.from_("attendance_proofs").upload(
        storage_path,
        image_bytes,
        {
            "content-type": "image/jpeg",
            "upsert": "false",
        },
    )

    return storage_path


def _extract_storage_path_from_url(value: str) -> str | None:
    markers = [
        "/storage/v1/object/public/attendance_proofs/",
        "/storage/v1/object/sign/attendance_proofs/",
        "/storage/v1/object/authenticated/attendance_proofs/",
    ]

    for marker in markers:
        index = value.find(marker)
        if index >= 0:
            start_index = index + len(marker)
            candidate = value[start_index:].split("?", 1)[0]
            if candidate:
                return candidate

    return None


def _normalize_attendance_storage_path(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None

    if raw.startswith("http://") or raw.startswith("https://"):
        extracted = _extract_storage_path_from_url(raw)
        if not extracted:
            return None
        raw = extracted

    normalized = raw.replace("\\", "/").lstrip("/")
    if normalized.lower().startswith("attendance_proofs/"):
        normalized = normalized[len("attendance_proofs/"):]

    if "/" not in normalized:
        return None

    return normalized


def _extract_signed_url_result(value) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        if isinstance(value.get("signedURL"), str):
            return value.get("signedURL")
        if isinstance(value.get("signedUrl"), str):
            return value.get("signedUrl")
        data = value.get("data")
        if isinstance(data, dict):
            if isinstance(data.get("signedURL"), str):
                return data.get("signedURL")
            if isinstance(data.get("signedUrl"), str):
                return data.get("signedUrl")

    data_attr = getattr(value, "data", None)
    if isinstance(data_attr, dict):
        if isinstance(data_attr.get("signedURL"), str):
            return data_attr.get("signedURL")
        if isinstance(data_attr.get("signedUrl"), str):
            return data_attr.get("signedUrl")

    return None


def _can_view_attendance_proof(uid: str, attendance_row: dict) -> bool:
    if str(attendance_row.get("user_id") or "") == str(uid):
        return True

    class_id = attendance_row.get("class_id")
    if not class_id:
        return False

    try:
        class_result = (
            supabase.table("classes")
            .select("owner_id")
            .eq("id", class_id)
            .single()
            .execute()
        )
        class_row = class_result.data if class_result and class_result.data else None
        if class_row and str(class_row.get("owner_id") or "") == str(uid):
            return True
    except Exception as err:
        logger.warning("Proof access check classes failed uid=%s class=%s err=%s", uid, class_id, str(err))

    try:
        member_result = (
            supabase.table("class_members")
            .select("role")
            .eq("class_id", class_id)
            .eq("user_id", uid)
            .maybe_single()
            .execute()
        )
        member_row = member_result.data if member_result and member_result.data else None
        role = str(member_row.get("role") or "").lower() if member_row else ""
        if role in {"owner", "instructor"}:
            return True
    except Exception as err:
        logger.warning("Proof access check class_members failed uid=%s class=%s err=%s", uid, class_id, str(err))

    return False

def _check_rate_limit(user_id: str):
    now = int(time.time())
    bucket = _RATE_STORE.get(user_id)
    if not bucket:
        _RATE_STORE[user_id] = {"count": 1, "reset_at": now + _RATE_LIMIT_WINDOW}
        return

    if now > bucket["reset_at"]:
        _RATE_STORE[user_id] = {"count": 1, "reset_at": now + _RATE_LIMIT_WINDOW}
        return

    if bucket["count"] >= _RATE_LIMIT_MAX:
        raise HTTPException(status_code=429, detail="Too many requests")

    bucket["count"] += 1


def _to_time(value) -> dt_time | None:
    if value is None:
        return None

    if isinstance(value, dt_time):
        return value

    if isinstance(value, str):
        raw_value = value.strip()
        if not raw_value:
            return None
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(raw_value, fmt).time()
            except ValueError:
                continue

    return None


def _get_class_schedule(class_id: str):
    class_result = (
        supabase.table("classes")
        .select("id, start_time, late_tolerance, timezone")
        .eq("id", class_id)
        .single()
        .execute()
    )
    return class_result.data if class_result and class_result.data else None


def _evaluate_time_window(class_data: dict):
    timezone_name = class_data.get("timezone") or "UTC"
    try:
        class_tz = ZoneInfo(timezone_name)
    except Exception:
        class_tz = timezone.utc

    start_time_obj = _to_time(class_data.get("start_time"))
    if start_time_obj is None:
        return {
            "status": "rejected",
            "reason": "Class schedule is incomplete. Please contact the instructor or admin.",
            "now_local": datetime.now(class_tz),
            "cutoff_local": None,
        }

    late_tolerance = class_data.get("late_tolerance")
    try:
        late_tolerance_minutes = int(late_tolerance or 0)
    except Exception:
        late_tolerance_minutes = 0

    now_local = datetime.now(class_tz)
    start_local = datetime.combine(now_local.date(), start_time_obj, tzinfo=class_tz)
    cutoff_local = start_local + timedelta(minutes=max(late_tolerance_minutes, 0))

    if now_local <= start_local:
        return {
            "status": "present",
            "reason": "Attendance accepted on time.",
            "now_local": now_local,
            "cutoff_local": cutoff_local,
        }

    if now_local <= cutoff_local:
        return {
            "status": "late",
            "reason": "Attendance accepted, but marked late.",
            "now_local": now_local,
            "cutoff_local": cutoff_local,
        }

    return {
        "status": "rejected",
        "reason": "Attendance time has passed the class late tolerance limit.",
        "now_local": now_local,
        "cutoff_local": cutoff_local,
    }


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _is_face_not_detected_error(error_text: str | None) -> bool:
    if not error_text:
        return False

    lower_text = error_text.lower()
    keywords = [
        "could not be detected",
        "no face detected",
        "face could not be detected",
        "enforce_detection",
        "no face",
    ]
    return any(keyword in lower_text for keyword in keywords)


def _friendly_face_processing_message(error_text: str | None) -> str:
    if _is_face_not_detected_error(error_text):
        return "Face not detected. Make sure your face is clearly visible, inside the frame, and well lit."
    return "Face photo could not be processed. Please retake it with a stable camera and better lighting."


@app.post("/register")
async def process_ai(
    request: Request,
    uid=Depends(get_user_id),
    hmac_ok: bool = Depends(verify_hmac_request),
):
    if not hmac_ok:
        raise HTTPException(status_code=401, detail="HMAC verification failed")

    _check_rate_limit(uid)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    image_front = payload.get("image_front")
    image_left = payload.get("image_left")
    image_right = payload.get("image_right")
    raw_capture_source = str(payload.get("capture_source") or "onboarding").strip().lower()
    capture_source = raw_capture_source if raw_capture_source in {"onboarding", "profile"} else "onboarding"

    # Backward compatibility: allow legacy array payload and map by order.
    if not (isinstance(image_front, str) and isinstance(image_left, str) and isinstance(image_right, str)):
        images = payload.get("image")
        if isinstance(images, list) and len(images) >= 3:
            image_front = images[0]
            image_left = images[1]
            image_right = images[2]

    if not image_front or not isinstance(image_front, str):
        raise HTTPException(status_code=400, detail="image_front is required")
    if not image_left or not isinstance(image_left, str):
        raise HTTPException(status_code=400, detail="image_left is required")
    if not image_right or not isinstance(image_right, str):
        raise HTTPException(status_code=400, detail="image_right is required")

    logger.info("Embedding request from user=%s for front/left/right", uid)

    normalized_register_paths: dict[str, str] = {}
    try:
        image_front_bytes = resize_base64_image_to_jpeg_bytes(image_front)
        image_left_bytes = resize_base64_image_to_jpeg_bytes(image_left)
        image_right_bytes = resize_base64_image_to_jpeg_bytes(image_right)

        image_front = base64.b64encode(image_front_bytes).decode('utf-8')
        image_left = base64.b64encode(image_left_bytes).decode('utf-8')
        image_right = base64.b64encode(image_right_bytes).decode('utf-8')

        normalized_register_paths = _save_register_normalized_images(
            image_front_bytes,
            image_left_bytes,
            image_right_bytes,
            uid,
            capture_source,
        )
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to normalize embedding image to {TARGET_WIDTH}x{TARGET_HEIGHT} (3:4 portrait): {str(err)}",
        )

    front_embedding = getFaceEmbedding(image_front)
    left_embedding = getFaceEmbedding(image_left)
    right_embedding = getFaceEmbedding(image_right)

    if isinstance(front_embedding, dict) and "error" in front_embedding:
        raise HTTPException(status_code=400, detail=f"Failed to process front face: {front_embedding['error']}")
    if isinstance(left_embedding, dict) and "error" in left_embedding:
        raise HTTPException(status_code=400, detail=f"Failed to process left face: {left_embedding['error']}")
    if isinstance(right_embedding, dict) and "error" in right_embedding:
        raise HTTPException(status_code=400, detail=f"Failed to process right face: {right_embedding['error']}")

    profile_data = {
            "id": uid,
            "email": payload.get("email", ""),
            "username": payload.get("username", ""),
            "gender": payload.get("gender", ""),
            "avatar_url": payload.get("avatar_url", ""),
    }
    
    supabase.table("profiles").upsert(profile_data).execute()

    embedding_saved = _save_profile_embeddings(uid, front_embedding, left_embedding, right_embedding)
    if not embedding_saved:
        logger.warning("Profile embeddings could not be saved for user=%s. Similarity checks may be unavailable.", uid)

    return JSONResponse(status_code=200, content={
        "status": "ok",
        "message": "Profile updated",
        "capture_source": capture_source,
        "normalized_embedding_images": normalized_register_paths,
        "normalized_resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
    })

@app.post("/attendence")
@app.post("/attendance")
@app.post("/start-precence")
async def start_precence(
    request: Request,
    uid=Depends(get_user_id),
    hmac_ok: bool = Depends(verify_hmac_request),
):
    if not hmac_ok:
        raise HTTPException(status_code=401, detail="HMAC verification failed")

    _check_rate_limit(uid)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    class_id = payload.get("class_id")
    image = payload.get("image")
    latitude = payload.get("latitude")
    longitude = payload.get("longitude")
    distance_meters = payload.get("distance_meters")

    if not class_id:
        raise HTTPException(status_code=400, detail="class_id is required")
    if not image or not isinstance(image, str):
        raise HTTPException(status_code=400, detail="image is required")

    class_data = None
    try:
        class_data = _get_class_schedule(class_id)
    except Exception as err:
        logger.warning("Failed to fetch class schedule class=%s err=%s", class_id, str(err))

    if not class_data:
        raise HTTPException(status_code=404, detail="Class not found or schedule unavailable")

    time_eval = _evaluate_time_window(class_data)

    try:
        resized_image_bytes = resize_base64_image_to_jpeg_bytes(image)
        resized_image_b64 = resize_base64_image_to_jpeg_base64(image)
    except Exception as err:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to normalize attendance image to {TARGET_WIDTH}x{TARGET_HEIGHT} (3:4 portrait): {str(err)}",
        )

    saved_image_path = None
    normalized_attendance_path = None
    attendance_photo_url = None
    if SAVE_ATTENDANCE_IMAGE:
        try:
            saved_image_path = _save_attendance_image_jpg(resized_image_bytes, uid, class_id)
            attendance_photo_url = _upload_attendance_image_to_storage(resized_image_bytes, uid, class_id)

            logger.info(
                "Attendance image saved user=%s class=%s path=%s photo_url=%s",
                uid,
                class_id,
                saved_image_path,
                attendance_photo_url,
            )
        except Exception as err:
            logger.warning("Failed to save attendance image user=%s class=%s err=%s", uid, class_id, str(err))
    else:
        logger.info("Attendance image saving disabled via SAVE_ATTENDANCE_IMAGE")

    try:
        normalized_attendance_path = _save_attendance_normalized_image(resized_image_bytes, uid, class_id)
    except Exception as err:
        logger.warning("Failed to save normalized attendance debug image user=%s class=%s err=%s", uid, class_id, str(err))

    quality_result = None
    quality_failed = False
    quality_failure_message = None

    if IMAGE_QUALITY_CHECK_ENABLED:
        quality_result = checkImageQuality(
            resized_image_b64,
            min_brightness=MIN_IMAGE_BRIGHTNESS,
            min_sharpness=MIN_IMAGE_SHARPNESS,
        )
        quality_error = quality_result.get("error") if isinstance(quality_result, dict) else None

        if quality_error:
            logger.warning(
                "Image quality check error user=%s class=%s err=%s",
                uid,
                class_id,
                quality_error,
            )
        else:
            too_dark = bool(quality_result.get("too_dark"))
            too_blurry = bool(quality_result.get("too_blurry"))
            if too_dark or too_blurry:
                quality_failed = True
                if too_dark and too_blurry:
                    quality_failure_message = "Attendance photo is too dark and blurry. Move to brighter lighting and hold your phone steady."
                elif too_dark:
                    quality_failure_message = "Attendance photo is too dark. Move to brighter lighting and retake the photo."
                else:
                    quality_failure_message = "Attendance photo is too blurry. Hold your phone steady and retake the photo."

    # Liveness detection (optional, configurable via LIVENESS_DETECTION_ENABLED)
    liveness_result = None
    is_live = None
    liveness_confidence = None
    liveness_error = None
    
    liveness_supported = True
    liveness_failed = False

    if LIVENESS_ENABLED:
        try:
            liveness_result = checkLiveness(resized_image_b64)
            liveness_supported = bool(liveness_result.get("supported", True))

            if not liveness_supported:
                logger.warning("Liveness anti-spoofing not supported by installed DeepFace version")

            if "error" in liveness_result and liveness_result["error"]:
                logger.warning("Liveness check failed user=%s class=%s error=%s", uid, class_id, liveness_result["error"])
                liveness_error = liveness_result["error"]
            else:
                is_live = liveness_result.get("is_alive", False)
                liveness_confidence = liveness_result.get("confidence", 0.0)
                logger.info("Liveness check result user=%s class=%s is_live=%s confidence=%.2f", uid, class_id, is_live, liveness_confidence)
                
                if is_live is False and liveness_confidence is not None and liveness_confidence < LIVENESS_CONFIDENCE_THRESHOLD:
                    liveness_failed = True
                    logger.warning("Liveness check flagged spoof user=%s class=%s confidence=%.2f", uid, class_id, liveness_confidence)
        except Exception as err:
            logger.warning("Liveness detection exception user=%s class=%s err=%s", uid, class_id, str(err))
            liveness_error = str(err)
    else:
        logger.info("Liveness detection disabled for user=%s class=%s", uid, class_id)

    face_embedding = None
    face_processing_error = None

    # Best-effort face processing only (do not fail attendance flow).
    try:
        embedding = getFaceEmbedding(resized_image_b64)
        if isinstance(embedding, dict) and "error" in embedding:
            logger.warning("Face processing warning user=%s class=%s error=%s", uid, class_id, embedding["error"])
            face_processing_error = embedding["error"]
        else:
            face_embedding = embedding
    except Exception as err:
        logger.warning("Face processing exception user=%s class=%s err=%s", uid, class_id, str(err))
        face_processing_error = str(err)

    profile_embeddings = _fetch_profile_embeddings(uid)

    similarity_values: list[float] = []
    if face_embedding and profile_embeddings:
        for index, profile_embedding in enumerate(profile_embeddings):
            try:
                similarity = cosineSimilarity(face_embedding, profile_embedding)
                if isinstance(similarity, (float, int)):
                    similarity_values.append(float(similarity))
            except Exception as err:
                logger.warning("Failed to compute similarity user=%s profile_embedding_index=%d err=%s", uid, index, str(err))

    max_similarity = max(similarity_values) if similarity_values else None

    # Determine overall attendance success/failure status
    attendance_success = True
    attendance_message = "Attendance accepted"
    attendance_status = "present"

    if quality_failed:
        attendance_success = False
        attendance_status = "rejected"
        attendance_message = quality_failure_message or "Attendance photo quality is too low. Retake the photo in better conditions."
    
    # Check face processing errors
    if face_processing_error:
        attendance_success = False
        attendance_status = "rejected"
        attendance_message = _friendly_face_processing_message(face_processing_error)

    if attendance_success and not face_embedding:
        attendance_success = False
        attendance_status = "rejected"
        attendance_message = "Face is not captured clearly yet. Please retake your attendance photo."
    
    # Check liveness (if enabled and not already failed)
    if attendance_success and LIVENESS_ENABLED:
        if liveness_failed and LIVENESS_FAIL_ACTION == "block":
            attendance_success = False
            attendance_message = "Face liveness verification failed. Use a real face (not photo/video) and try again."
            attendance_status = "rejected"
    
    if attendance_success and face_embedding:
        if similarity_values:
            has_sufficient_match = max_similarity is not None and max_similarity >= MATCH_SIMILARITY_THRESHOLD
            if not has_sufficient_match:
                attendance_success = False
                attendance_status = "rejected"
                attendance_message = "Face does not match your profile data. Try a straighter pose with brighter lighting."
        else:
            logger.warning("No profile embeddings found for user=%s; similarity-based check skipped", uid)

    if attendance_success:
        if time_eval["status"] == "rejected":
            attendance_success = False
            attendance_status = "rejected"
            attendance_message = time_eval["reason"]
        elif time_eval["status"] == "late":
            attendance_status = "late"
            attendance_message = "Attendance accepted, but your status is late."
        else:
            attendance_status = "present"
            attendance_message = "Attendance accepted."

    attendance_recorded = False
    attendance_record_error = None
    try:
        attendance_payload = {
            "class_id": class_id,
            "user_id": uid,
            "status": attendance_status,
            "presence_at": datetime.now(timezone.utc).isoformat(),
            "similarity_score": _safe_float(max_similarity),
            "user_lat": _safe_float(latitude),
            "user_lon": _safe_float(longitude),
            "rejection_reason": attendance_message if attendance_status == "rejected" else None,
            "photo_url": attendance_photo_url,
        }
        insert_result = supabase.table("attendances").insert(attendance_payload).execute()
        if not insert_result or not getattr(insert_result, "data", None):
            raise RuntimeError("Insert attendances did not return row data")

        attendance_recorded = True
    except Exception as err:
        attendance_record_error = str(err)
        logger.warning("Failed to insert attendances row user=%s class=%s err=%s", uid, class_id, attendance_record_error)
        raise HTTPException(status_code=500, detail="Failed to save attendance data to database")

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "attendance_success": attendance_success,
            "attendance_status": attendance_status,
            "attendance_message": attendance_message,
            "highest_similarity": _safe_float(max_similarity),
            "normalized_attendance_image": normalized_attendance_path,
            "normalized_resolution": f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
            "image_quality": quality_result,
        },
    )

@app.post('/hmac/issue')
async def hmac_issue(uid=Depends(get_user_id)):
    return issue_hmac_key(uid)


@app.post('/attendance/proof-url')
async def attendance_proof_url(
    request: Request,
    uid=Depends(get_user_id),
    hmac_ok: bool = Depends(verify_hmac_request),
):
    if not hmac_ok:
        raise HTTPException(status_code=401, detail="HMAC verification failed")

    _check_rate_limit(uid)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    attendance_id = payload.get("attendance_id")
    requested_storage_path = payload.get("storage_path")

    if not attendance_id:
        raise HTTPException(status_code=400, detail="attendance_id is required")

    attendance_result = (
        supabase.table("attendances")
        .select("id, class_id, user_id, photo_url")
        .eq("id", attendance_id)
        .single()
        .execute()
    )

    attendance_row = attendance_result.data if attendance_result and attendance_result.data else None
    if not attendance_row:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    if not _can_view_attendance_proof(uid, attendance_row):
        raise HTTPException(status_code=403, detail="Not allowed to view this attendance proof")

    normalized_path = _normalize_attendance_storage_path(requested_storage_path)
    if not normalized_path:
        normalized_path = _normalize_attendance_storage_path(attendance_row.get("photo_url"))

    if not normalized_path:
        raise HTTPException(status_code=404, detail="Attendance proof path not found")

    try:
        signed_result = supabase.storage.from_("attendance_proofs").create_signed_url(normalized_path, 3600)
        signed_url = _extract_signed_url_result(signed_result)
        if not signed_url:
            raise RuntimeError("create_signed_url returned empty URL")

        return JSONResponse(
            status_code=200,
            content={
                "status": "ok",
                "attendance_id": attendance_id,
                "storage_path": normalized_path,
                "signed_url": signed_url,
            },
        )
    except Exception as err:
        logger.warning(
            "Failed to create attendance proof signed URL uid=%s attendance_id=%s path=%s err=%s",
            uid,
            attendance_id,
            normalized_path,
            str(err),
        )
        raise HTTPException(status_code=500, detail="Failed to prepare attendance proof URL")


def secrets_token_short() -> str:
    import secrets
    return secrets.token_urlsafe(6)
