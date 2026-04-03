import base64
import numpy as np
import cv2

from deepface import DeepFace


TARGET_WIDTH = 384
TARGET_HEIGHT = 512


def _decode_base64_image(image: str):
    raw_value = image.strip() if isinstance(image, str) else image
    if isinstance(raw_value, str) and ',' in raw_value and raw_value.lower().startswith('data:'):
        raw_value = raw_value.split(',', 1)[1]

    padded_value = raw_value + '=' * (-len(raw_value) % 4)
    contents = base64.b64decode(padded_value)
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _center_crop_to_ratio(image_mat, width: int, height: int):
    if image_mat is None:
        return None

    src_h, src_w = image_mat.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return None

    target_ratio = width / height
    source_ratio = src_w / src_h

    if source_ratio > target_ratio:
        crop_w = int(src_h * target_ratio)
        x0 = max((src_w - crop_w) // 2, 0)
        return image_mat[:, x0:x0 + crop_w]

    crop_h = int(src_w / target_ratio)
    y0 = max((src_h - crop_h) // 2, 0)
    return image_mat[y0:y0 + crop_h, :]


def _resize_to_target(image_mat, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT):
    if image_mat is None:
        return None

    cropped = _center_crop_to_ratio(image_mat, width=width, height=height)
    if cropped is None:
        return None

    src_h, src_w = cropped.shape[:2]
    interpolation = cv2.INTER_AREA if src_w >= width and src_h >= height else cv2.INTER_LINEAR
    return cv2.resize(cropped, (width, height), interpolation=interpolation)


def resize_base64_image_to_jpeg_bytes(image: str, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT) -> bytes:
    decoded = _decode_base64_image(image)
    if decoded is None:
        raise ValueError("Failed to decode image")

    resized = _resize_to_target(decoded, width=width, height=height)
    ok, encoded = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise ValueError("Failed to encode resized image")

    return encoded.tobytes()


def resize_base64_image_to_jpeg_base64(image: str, width: int = TARGET_WIDTH, height: int = TARGET_HEIGHT) -> str:
    encoded_bytes = resize_base64_image_to_jpeg_bytes(image, width=width, height=height)
    return base64.b64encode(encoded_bytes).decode('utf-8')


def checkImageQuality(image: str, min_brightness: float = 55.0, min_sharpness: float = 45.0) -> dict:
    try:
        img = _decode_base64_image(image)
        if img is None:
            return {
                "is_acceptable": False,
                "too_dark": None,
                "too_blurry": None,
                "brightness_score": None,
                "sharpness_score": None,
                "error": "Failed to decode image",
            }

        img = _resize_to_target(img)
        if img is None:
            return {
                "is_acceptable": False,
                "too_dark": None,
                "too_blurry": None,
                "brightness_score": None,
                "sharpness_score": None,
                "error": "Failed to normalize image",
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_score = float(np.mean(gray))
        sharpness_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        too_dark = brightness_score < float(min_brightness)
        too_blurry = sharpness_score < float(min_sharpness)

        return {
            "is_acceptable": not (too_dark or too_blurry),
            "too_dark": too_dark,
            "too_blurry": too_blurry,
            "brightness_score": brightness_score,
            "sharpness_score": sharpness_score,
            "error": None,
        }
    except Exception as e:
        return {
            "is_acceptable": False,
            "too_dark": None,
            "too_blurry": None,
            "brightness_score": None,
            "sharpness_score": None,
            "error": str(e),
        }

def getFaceEmbedding(image: str):
    try:
        img = _decode_base64_image(image)
        if img is None:
            return {"error": "Failed to decode image"}

        img = _resize_to_target(img)

        embeddings_data = DeepFace.represent(
            img_path=img, 
            model_name="ArcFace",          
            detector_backend="retinaface", 
            enforce_detection=True
        )
        # embeddings_data = DeepFace.represent(
        #     img_path=img, 
        #     model_name="Facenet", # Facenet lumayan oke, kalau masih berat ganti "OpenFace"
        #     detector_backend="opencv", # <-- INI KUNCI BIAR NGGAK BERAT
        #     enforce_detection=True
        # )
        
        # Ambil vektor embedding (biasanya 128 atau 512 dimensi)
        face_embedding = embeddings_data[0]["embedding"]
        
        return face_embedding


    except Exception as e:
        return {"error": str(e)}


def cosineSimilarity(embedding1: list, embedding2: list) -> float:
    """
    Compute cosine similarity between two embeddings.
    Range: -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)
    """
    if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
        return 0.0

    arr1 = np.array(embedding1)
    arr2 = np.array(embedding2)

    dot_product = np.dot(arr1, arr2)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)

    denominator = norm1 * norm2
    if denominator == 0:
        return 0.0

    return float(dot_product / denominator)


def checkLiveness(image: str) -> dict:
    """
    Perform liveness detection on a face image using DeepFace.
    Returns: {
        'is_alive': bool,
        'confidence': float (0-1),
        'model_output': str (Real/Fake),
        'error': str (if failed)
    }
    """
    try:
        img = _decode_base64_image(image)

        if img is None:
            return {"error": "Failed to decode image", "supported": True}

        img = _resize_to_target(img)

        try:
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend="retinaface",
                enforce_detection=True,
                anti_spoofing=True,
            )
        except TypeError:
            return {
                "is_alive": None,
                "confidence": None,
                "model_output": "unsupported",
                "supported": False,
                "error": None,
            }

        if not faces:
            return {"error": "No face detected in image", "supported": True}

        primary_face = faces[0]
        anti_spoof_supported = "is_real" in primary_face or "antispoof_score" in primary_face

        if not anti_spoof_supported:
            return {
                "is_alive": None,
                "confidence": None,
                "model_output": "unsupported",
                "supported": False,
                "error": None,
            }

        is_real = bool(primary_face.get("is_real", False))
        confidence = primary_face.get("antispoof_score", 0.0)

        return {
            "is_alive": is_real,
            "confidence": float(confidence),
            "model_output": "Real" if is_real else "Fake",
            "supported": True,
            "error": None,
        }

    except Exception as e:
        return {"error": str(e), "supported": True}
