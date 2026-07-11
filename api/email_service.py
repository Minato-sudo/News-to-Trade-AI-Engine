"""
api/email_service.py
====================
Sends OTP emails via Gmail SMTP for password reset.
Requires SMTP_EMAIL and SMTP_APP_PASSWORD in .env
"""

import smtplib, random, string, time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# In-memory OTP store: {email: (otp, expires_at)}
# In production replace with Redis
_otp_store: dict[str, tuple[str, float]] = {}

OTP_TTL_SECONDS = 600  # 10 minutes


def generate_otp(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


def store_otp(email: str, otp: str):
    _otp_store[email.lower()] = (otp, time.time() + OTP_TTL_SECONDS)


def verify_otp(email: str, otp: str) -> bool:
    key = email.lower()
    if key not in _otp_store:
        return False
    stored_otp, expires_at = _otp_store[key]
    if time.time() > expires_at:
        del _otp_store[key]
        return False
    if stored_otp != otp:
        return False
    del _otp_store[key]   # one-time use
    return True


def send_otp_email(to_email: str, otp: str, username: str = "") -> bool:
    """Send OTP via Gmail SMTP. Returns True on success."""
    from config import settings

    smtp_email    = getattr(settings, "SMTP_EMAIL", "")
    smtp_password = getattr(settings, "SMTP_APP_PASSWORD", "")

    if not smtp_email or not smtp_password:
        logger.warning("[EMAIL] SMTP not configured — OTP printed to console instead.")
        print(f"\n{'='*50}")
        print(f"  PASSWORD RESET OTP for {to_email}")
        print(f"  Code: {otp}  (valid 10 min)")
        print(f"{'='*50}\n")
        return True   # allow flow to continue in dev mode

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Your Password Reset Code — Storyline-to-Signal"
        msg["From"]    = smtp_email
        msg["To"]      = to_email

        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#0f0f1a;color:#e0e0e0;padding:40px;">
          <div style="max-width:480px;margin:auto;background:#1a1a2e;border-radius:12px;padding:32px;border:1px solid #2d2d4e;">
            <div style="text-align:center;margin-bottom:24px;">
              <span style="font-size:40px;">⚡</span>
              <h2 style="color:#7c6af7;margin:8px 0;">Storyline-to-Signal</h2>
            </div>
            <p>Hi <strong>{username or to_email}</strong>,</p>
            <p>Your password reset code is:</p>
            <div style="text-align:center;margin:28px 0;">
              <span style="font-size:42px;font-weight:bold;letter-spacing:12px;color:#7c6af7;
                           background:#0f0f1a;padding:16px 24px;border-radius:8px;display:inline-block;">
                {otp}
              </span>
            </div>
            <p style="color:#888;font-size:13px;">This code expires in <strong>10 minutes</strong>.<br>
            If you didn't request this, ignore this email.</p>
          </div>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_email, smtp_password)
            server.sendmail(smtp_email, to_email, msg.as_string())

        logger.info(f"[EMAIL] OTP sent to {to_email}")
        return True

    except Exception as e:
        logger.error(f"[EMAIL] Failed to send OTP to {to_email}: {e}")
        return False
