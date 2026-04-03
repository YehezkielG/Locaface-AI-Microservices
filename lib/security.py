import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Dict, Optional

import httpx
from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

load_dotenv()

security = HTTPBearer()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    # Fail fast at runtime with clear message.
    # These must be set on the server (never ship secrets to mobile app).
    print('WARNING: SUPABASE_URL / SUPABASE_ANON_KEY not set. Auth verification will fail.')


async def _verify_supabase_access_token(access_token: str) -> dict:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail='Server misconfigured: missing Supabase env')

    url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/user"
    headers = {
        'apikey': SUPABASE_ANON_KEY,
        'Authorization': f'Bearer {access_token}',
    }

    async with httpx.AsyncClient(timeout=7.0) as client:
        res = await client.get(url, headers=headers)

    if res.status_code != 200:
        raise HTTPException(status_code=401, detail='Token palsu / expired')

    return res.json()


async def get_user_id(res: HTTPAuthorizationCredentials = Depends(security)):
    token = res.credentials
    user = await _verify_supabase_access_token(token)
    uid = user.get('id')
    if not uid:
        raise HTTPException(status_code=401, detail='Invalid user payload')
    return uid


# --- HMAC signing (anti-tamper + anti-replay) ---

HMAC_TTL_SECONDS = int(os.getenv('HMAC_TTL_SECONDS', '900'))  # 15 minutes
HMAC_ALLOWED_SKEW_SECONDS = int(os.getenv('HMAC_ALLOWED_SKEW_SECONDS', '300'))  # 5 minutes
HMAC_MASTER_SECRET = os.getenv('HMAC_MASTER_SECRET', 'locaface-dev-hmac-master-secret')


class _HmacKeyEntry(dict):
    pass


_HMAC_KEYS: Dict[str, _HmacKeyEntry] = {}
_NONCES: Dict[str, int] = {}


def _now() -> int:
    return int(time.time())


def _cleanup() -> None:
    now = _now()
    # cleanup expired keys
    expired_keys = [k for k, v in _HMAC_KEYS.items() if v.get('expires_at', 0) <= now]
    for k in expired_keys:
        _HMAC_KEYS.pop(k, None)

    # cleanup used nonces
    expired_nonces = [n for n, exp in _NONCES.items() if exp <= now]
    for n in expired_nonces:
        _NONCES.pop(n, None)


def issue_hmac_key(user_id: str) -> dict:
    _cleanup()
    expires_at = _now() + HMAC_TTL_SECONDS
    key_nonce = secrets.token_urlsafe(12)

    raw_payload = f"{user_id}.{expires_at}.{key_nonce}".encode('utf-8')
    key_id = base64.urlsafe_b64encode(raw_payload).decode('utf-8').rstrip('=')
    secret = hmac.new(HMAC_MASTER_SECRET.encode('utf-8'), key_id.encode('utf-8'), hashlib.sha256).digest()

    # Keep temporary backward compatibility cache for sessions created before this change.
    # Verification now primarily uses stateless derivation from key_id.

    _HMAC_KEYS[key_id] = _HmacKeyEntry({
        'user_id': user_id,
        'secret': secret,
        'expires_at': expires_at,
    })

    return {
        'key_id': key_id,
        'secret_b64': base64.b64encode(secret).decode('utf-8'),
        'expires_at': expires_at,
    }


def _derive_stateless_hmac_key(key_id: str, user_id: str):
    try:
        padding = '=' * (-len(key_id) % 4)
        decoded = base64.urlsafe_b64decode((key_id + padding).encode('utf-8')).decode('utf-8')
        key_user_id, expires_at_raw, _nonce = decoded.split('.', 2)
        expires_at = int(expires_at_raw)
    except Exception:
        return None, None

    if key_user_id != user_id:
        return None, None

    secret = hmac.new(HMAC_MASTER_SECRET.encode('utf-8'), key_id.encode('utf-8'), hashlib.sha256).digest()
    return secret, expires_at


async def verify_hmac_request(
    request: Request,
    user_id: str = Depends(get_user_id),
    x_hmac_key_id: Optional[str] = Header(None),
    x_hmac_timestamp: Optional[str] = Header(None),
    x_hmac_nonce: Optional[str] = Header(None),
    x_hmac_signature: Optional[str] = Header(None),
    x_body_sha256: Optional[str] = Header(None),
):
    _cleanup()

    print("user id for HMAC verification:", x_hmac_key_id)

    if not x_hmac_key_id or not x_hmac_timestamp or not x_hmac_nonce or not x_hmac_signature or not x_body_sha256:
        raise HTTPException(status_code=401, detail='HMAC headers missing')

    try:
        ts = int(x_hmac_timestamp)
    except ValueError:
        raise HTTPException(status_code=401, detail='Invalid timestamp')

    now = _now()
    if abs(now - ts) > HMAC_ALLOWED_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail='Request expired')

    if x_hmac_nonce in _NONCES:
        raise HTTPException(status_code=401, detail='Replay detected')

    entry = _HMAC_KEYS.get(x_hmac_key_id)
    print("HMAC entry found:", entry)

    if entry:
        if entry.get('user_id') != user_id:
            raise HTTPException(status_code=401, detail='HMAC key mismatch')
        if entry.get('expires_at', 0) <= now:
            raise HTTPException(status_code=401, detail='HMAC key expired')
        signing_secret = entry['secret']
    else:
        signing_secret, expires_at = _derive_stateless_hmac_key(x_hmac_key_id, user_id)
        if not signing_secret:
            raise HTTPException(status_code=401, detail='Unknown HMAC key')
        if expires_at <= now:
            raise HTTPException(status_code=401, detail='HMAC key expired')

    body_bytes = await request.body()
    body_hash = hashlib.sha256(body_bytes or b'').hexdigest()
    if not hmac.compare_digest(body_hash, x_body_sha256.lower()):
        raise HTTPException(status_code=401, detail='Body hash mismatch')

    canonical = f"{request.method.upper()}\n{request.url.path}\n{ts}\n{x_hmac_nonce}\n{body_hash}"
    expected_sig = base64.b64encode(hmac.new(signing_secret, canonical.encode('utf-8'), hashlib.sha256).digest()).decode('utf-8')

    if not hmac.compare_digest(expected_sig, x_hmac_signature):
        raise HTTPException(status_code=401, detail='Bad signature')

    # record nonce as used (expires when skew window passes)
    _NONCES[x_hmac_nonce] = now + HMAC_ALLOWED_SKEW_SECONDS

    return True