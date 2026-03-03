"""
Unit tests for app.auth
------------------------
Tests JWT creation/verification and password hashing — no DB, no HTTP.
"""

from datetime import timedelta

import pytest
from jose import jwt

from app.auth import create_access_token, hash_password, verify_password
from app.config import settings


@pytest.mark.unit
class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = hash_password("secret123")
        assert hashed != "secret123"

    def test_verify_correct_password(self):
        hashed = hash_password("correct-horse-battery-staple")
        assert verify_password("correct-horse-battery-staple", hashed) is True

    def test_reject_wrong_password(self):
        hashed = hash_password("correct-password")
        assert verify_password("wrong-password", hashed) is False

    def test_hash_is_deterministically_different_each_call(self):
        # bcrypt uses random salt — same input → different hash every time
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert h1 != h2

    def test_verify_still_works_despite_different_hashes(self):
        h1 = hash_password("same-password")
        h2 = hash_password("same-password")
        assert verify_password("same-password", h1) is True
        assert verify_password("same-password", h2) is True


@pytest.mark.unit
class TestCreateAccessToken:
    def _decode(self, token: str) -> dict:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])

    def test_token_contains_sub_claim(self):
        token = create_access_token({"sub": "admin"})
        payload = self._decode(token)
        assert payload["sub"] == "admin"

    def test_token_contains_exp_claim(self):
        token = create_access_token({"sub": "admin"})
        payload = self._decode(token)
        assert "exp" in payload

    def test_custom_expiry_delta(self):
        import time
        token = create_access_token({"sub": "admin"}, expires_delta=timedelta(seconds=5))
        payload = self._decode(token)
        # exp should be roughly now + 5 seconds (allow 2 second window)
        assert payload["exp"] - time.time() < 7

    def test_algorithm_is_hs256(self):
        token = create_access_token({"sub": "admin"})
        header = jwt.get_unverified_header(token)
        assert header["alg"] == "HS256"

    def test_token_invalid_with_wrong_secret(self):
        from jose import JWTError
        token = create_access_token({"sub": "admin"})
        with pytest.raises(JWTError):
            jwt.decode(token, "wrong-secret", algorithms=["HS256"])

    def test_extra_claims_preserved(self):
        token = create_access_token({"sub": "admin", "role": "superuser"})
        payload = self._decode(token)
        assert payload["role"] == "superuser"
