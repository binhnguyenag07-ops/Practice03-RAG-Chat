"""Firebase Email/Password authentication using the official Identity Toolkit REST API."""
import requests

API_ROOT = "https://identitytoolkit.googleapis.com/v1/accounts"


class FirebaseAuthError(Exception):
    pass


ERROR_MESSAGES = {
    "EMAIL_EXISTS": "This email is already registered.",
    "OPERATION_NOT_ALLOWED": "Enable Email/Password in Firebase Authentication settings.",
    "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please try again later.",
    "EMAIL_NOT_FOUND": "Invalid email or password.",
    "INVALID_PASSWORD": "Invalid email or password.",
    "INVALID_LOGIN_CREDENTIALS": "Invalid email or password.",
    "USER_DISABLED": "This account is disabled.",
    "WEAK_PASSWORD": "Password is too weak (use at least six characters).",
    "INVALID_EMAIL": "Invalid email address.",
}


def _authenticate(api_key: str, action: str, email: str, password: str) -> dict:
    if not api_key:
        raise FirebaseAuthError("FIREBASE_API_KEY is missing from .env.")
    if not email.strip() or not password:
        raise FirebaseAuthError("Email and password are required.")
    if action == "signUp" and len(password) < 6:
        raise FirebaseAuthError("Password must contain at least six characters.")

    try:
        response = requests.post(
            f"{API_ROOT}:{action}",
            params={"key": api_key},
            json={"email": email.strip(), "password": password, "returnSecureToken": True},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise FirebaseAuthError("Could not connect to Firebase Authentication.") from exc

    if not response.ok:
        try:
            code = response.json()["error"]["message"].split(" : ")[0]
        except (KeyError, ValueError, TypeError):
            code = f"HTTP_{response.status_code}"
        message = ERROR_MESSAGES.get(code, f"Firebase error: {code}")
        raise FirebaseAuthError(message)

    result = response.json()
    return {"email": result["email"], "uid": result["localId"], "id_token": result["idToken"]}


def sign_up(api_key: str, email: str, password: str) -> dict:
    return _authenticate(api_key, "signUp", email, password)


def sign_in(api_key: str, email: str, password: str) -> dict:
    return _authenticate(api_key, "signInWithPassword", email, password)
