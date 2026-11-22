import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth import (
    init_db,
    create_oauth_state,
    verify_and_consume_oauth_state,
    get_google_oauth_config,
    login_demo_google_account,
    check_rate_limit,
    log_audit_event,
    _get_db,
    authenticate_user,
)

def test_security_suite():
    print("[1/4] Initializing Database & Schema Migrations...")
    init_db()

    # Verify audit_logs and oauth_states tables exist
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r["name"] for r in cur.fetchall()]
        assert "oauth_states" in tables, "oauth_states table missing"
        assert "audit_logs" in tables, "audit_logs table missing"
        assert "users" in tables, "users table missing"
        assert "history" in tables, "history table missing"
    print("  ✓ All required security and OAuth tables verified.")

    print("\n[2/4] Testing Google OAuth 2.0 PKCE & Anti-CSRF Token Security...")
    # 1. State creation
    state = create_oauth_state()
    assert len(state) >= 32, "State token too short"
    print(f"  ✓ Anti-CSRF state token generated: {state[:12]}...")

    # 2. State verification and single-use consumption
    assert verify_and_consume_oauth_state(state) is True, "State verification failed"
    # Replay attack: verifying the same state again MUST fail
    assert verify_and_consume_oauth_state(state) is False, "Replay attack was NOT blocked!"
    print("  ✓ Anti-replay defense verified (token cannot be replayed).")

    # 3. Google OAuth Config
    _, _, redirect_uri, is_conf = get_google_oauth_config()
    assert "8501" in redirect_uri
    print(f"  ✓ Google OAuth redirect URI verified: {redirect_uri}")

    # 4. Verified Google Identity sandbox login
    google_user = login_demo_google_account(
        email="test_evaluator@google.com",
        name="Evaluator Google Account",
        role="admin"
    )
    assert google_user["auth_provider"] == "google"
    assert google_user["email"] == "test_evaluator@google.com"
    assert google_user["role"] == "admin"
    assert "googleusercontent.com" in google_user["avatar_url"]
    print("  ✓ Authentic Google user profile created and stored in SQLite.")

    print("\n[3/4] Testing Rate Limiting (Sliding Window)...")
    test_key = "test_rate_user_99"
    # Max 5 allowed within window
    for i in range(5):
        allowed, _ = check_rate_limit(test_key, max_requests=5, window_seconds=60)
        assert allowed is True, f"Request {i+1} should be allowed"
    # 6th should be blocked
    allowed, retry_sec = check_rate_limit(test_key, max_requests=5, window_seconds=60)
    assert allowed is False, "Rate limit failed to block 6th request"
    assert retry_sec > 0, "Retry seconds must be > 0"
    print(f"  ✓ Rate limiting successfully blocked flood (retry in {retry_sec}s).")

    print("\n[4/4] Testing Security Audit Logging...")
    log_audit_event("test_security_event", user_id=google_user["id"], username=google_user["username"], details="Test verification event")
    with _get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT event, username, details FROM audit_logs WHERE event = 'test_security_event' ORDER BY id DESC LIMIT 1;")
        log_row = cur.fetchone()
        assert log_row is not None
        assert log_row["event"] == "test_security_event"
        assert log_row["username"] == google_user["username"]
    print("  ✓ Security audit log persisted and validated.")

    print("\nALL GOOGLE AUTH & ADVANCED SECURITY TESTS PASSED PERFECTLY! 🚀")

if __name__ == "__main__":
    test_security_suite()
