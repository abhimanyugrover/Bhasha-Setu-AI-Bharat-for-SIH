import os
import sys
import time
import threading

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auth import (
    init_db,
    register_user,
    authenticate_user,
    save_user_history,
    get_user_history,
    clear_user_history,
    MAX_FAILED_ATTEMPTS,
)
from pipeline.concurrency import (
    acquire_dubbing_slot,
    get_queue_status,
    MAX_CONCURRENT_JOBS,
)

def test_auth():
    print("[1/2] Testing Auth & Multi-User Security...")
    init_db()

    # 1. Test Admin Login
    admin_user, err = authenticate_user("admin", "Admin@BhashaSetu2026!")
    assert admin_user is not None, f"Admin authentication failed: {err}"
    assert admin_user["role"] == "admin"
    print("  ✓ Admin authentication succeeded")

    # 2. Test User Registration
    test_user = "test_evaluator_01"
    test_pass = "SecurePass123!"
    ok, msg = register_user(test_user, test_pass)
    # Could already exist if rerun
    if not ok and "already taken" in msg:
        print("  ✓ User already exists, proceeding to authenticate")
    else:
        assert ok, f"Registration failed: {msg}"
        print("  ✓ User registration succeeded")

    # 3. Test Successful Login
    u, err = authenticate_user(test_user, test_pass)
    assert u is not None, f"Login failed: {err}"
    assert u["username"] == test_user
    user1_id = u["id"]
    print(f"  ✓ User login verified (user_id={user1_id})")

    # 4. Test Second User for Multi-User Isolation
    test_user2 = "test_evaluator_02"
    test_pass2 = "AnotherPass456!"
    register_user(test_user2, test_pass2)
    u2, _ = authenticate_user(test_user2, test_pass2)
    assert u2 is not None
    user2_id = u2["id"]
    print(f"  ✓ User 2 login verified (user_id={user2_id})")

    # 5. Test History Isolation
    save_user_history(user1_id, {
        "job_id": "job_user1_abc",
        "language": "Hindi",
        "source_lang": "English",
        "output_path": "output/user1.mp4",
        "transcript_len": 120,
    })
    save_user_history(user2_id, {
        "job_id": "job_user2_xyz",
        "language": "Tamil",
        "source_lang": "English",
        "output_path": "output/user2.mp4",
        "transcript_len": 240,
    })

    h1 = get_user_history(user1_id, is_admin=False)
    h2 = get_user_history(user2_id, is_admin=False)
    h_admin = get_user_history(admin_user["id"], is_admin=True)

    user1_jobs = [item["job_id"] for item in h1]
    user2_jobs = [item["job_id"] for item in h2]
    admin_jobs = [item["job_id"] for item in h_admin]

    assert "job_user1_abc" in user1_jobs
    assert "job_user2_xyz" not in user1_jobs, "User 1 should not see User 2's jobs!"
    assert "job_user2_xyz" in user2_jobs
    assert "job_user1_abc" not in user2_jobs, "User 2 should not see User 1's jobs!"
    assert "job_user1_abc" in admin_jobs and "job_user2_xyz" in admin_jobs, "Admin should see all jobs"
    print("  ✓ History isolation verified across concurrent users & admin access")

    # 6. Test Brute-Force Lockout
    lockout_user = "lockout_target"
    register_user(lockout_user, "ValidPass123!")
    for i in range(MAX_FAILED_ATTEMPTS):
        u, err = authenticate_user(lockout_user, "WrongPassword!")
        assert u is None
    # Next attempt should report account locked
    u, err = authenticate_user(lockout_user, "ValidPass123!")
    assert u is None
    assert "locked" in (err or "").lower(), f"Expected lockout message, got: {err}"
    print("  ✓ Brute-force protection verified: account locked after 5 failed attempts")


def test_concurrency():
    print("[2/2] Testing Concurrency Semaphore & Queuing...")
    status_initial = get_queue_status()
    print(f"  Initial queue status: {status_initial}")
    assert status_initial["active_jobs"] == 0

    results = []
    threads = []
    num_concurrent_workers = 6  # More than MAX_CONCURRENT_JOBS (3)

    def worker(worker_id):
        cb_msgs = []
        def _cb(msg):
            cb_msgs.append(msg)
        start_t = time.time()
        with acquire_dubbing_slot(job_id=f"sim_job_{worker_id}", status_cb=_cb):
            # Check queue status while holding slot
            q = get_queue_status()
            assert q["active_jobs"] <= MAX_CONCURRENT_JOBS, f"Active jobs {q['active_jobs']} exceeded max {MAX_CONCURRENT_JOBS}!"
            time.sleep(0.3)  # simulate workload
        elapsed = time.time() - start_t
        results.append({
            "worker_id": worker_id,
            "elapsed": elapsed,
            "cb_msgs": cb_msgs,
        })

    for i in range(num_concurrent_workers):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    status_final = get_queue_status()
    assert status_final["active_jobs"] == 0
    assert status_final["queued_jobs"] == 0
    assert len(results) == num_concurrent_workers

    # Workers launched after the first 3 should have received queue waiting notifications
    queued_workers = [r for r in results if any("Queue" in m or "waiting" in m for m in r["cb_msgs"])]
    print(f"  ✓ {len(results)} multi-user jobs processed safely.")
    print(f"  ✓ {len(queued_workers)} jobs waited gracefully in queue without system crash.")
    print("  ✓ Final active jobs returned to 0.")

if __name__ == "__main__":
    test_auth()
    test_concurrency()
    print("\nALL SECURITY & CONCURRENCY TESTS PASSED PERFECTLY!")
