# Privacy Policy — Bhasha Setu (AI Bharat)

**Last Updated:** November 21, 2026

Bhasha Setu ("we", "our", or "us") is dedicated to safeguarding user privacy and personal data. This Privacy Policy outlines our data handling practices for our multilingual video dubbing platform.

---

### 1. Information We Collect
- **Account Information:** Name, username, email address, password hash (PBKDF2-HMAC-SHA256), and authentication metadata.
- **OAuth Identity Data:** If you authenticate via Google Sign-In, we receive your verified email, display name, and avatar URL via Google Identity Services. We do not receive or store your Google account password.
- **Audio & Video Content:** Video files uploaded or provided via public URLs for transcription, translation, and speech synthesis.
- **Technical & Security Logs:** IP address (for brute-force rate-limiting only), session timestamps, anti-CSRF tokens, and security audit events.

---

### 2. How We Use Information
- To authenticate authorized users and maintain secure user sessions.
- To execute AI dubbing, transcription (AWS Transcribe / Sarvam AI), translation, and voice synthesis (Sarvam Bulbul:v1, Edge-TTS, AWS Polly).
- To isolate personal dubbing history so users can securely access their previous outputs.
- To protect the service against brute-force attacks, credential stuffing, and unauthorized access.

---

### 3. Data Storage & Security
- **Credentials:** Passwords are never stored in plain text; they are hashed with PBKDF2-HMAC-SHA256 using 100,000 iterations and unique 16-byte cryptographic salts.
- **Database:** Stored in thread-safe SQLite WAL mode or encrypted cloud databases (Supabase / AWS PostgreSQL).
- **Session Protection:** Inactivity timeout enforces automatic logout after 60 minutes.
- **Temporary Video Storage:** Staged input files are pruned and deleted immediately after pipeline completion.

---

### 4. Cookies & Tracking
We only use strictly necessary technical session cookies to maintain your authenticated state and prevent Cross-Site Request Forgery (CSRF). We do not deploy third-party advertising or cross-site tracking cookies.

---

### 5. Contact & Inquiries
For data deletion requests or privacy inquiries, contact: `privacy@bhashasetu.gov.in`
