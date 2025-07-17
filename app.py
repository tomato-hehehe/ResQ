# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import os
import smtplib
from email.message import EmailMessage
import requests
import time
from datetime import datetime
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import uuid # For unique alert IDs
import threading # For background alert cancellation logic
import json # For pretty printing API response in client

# Load environment variables from .env file
load_dotenv()

# --- Configuration (UPDATE THESE) ---
# YOLO Model Path
MODEL_PATH = "runs/detect/train/weights/best.pt" # Make sure this path is correct
                                                  # e.g., if app.py is in the root, and best.pt is in subfolder, adjust.

# Alerting Thresholds
CRASH_CONF_THRESHOLD = 0.5  # Confidence score for a crash detection
MIN_CRASH_DETECTIONS_REQUIRED = 3 # Number of consecutive frames with crash to trigger alert

# Emergency Contacts & Alert Settings
# SMS Gateway (Fast2SMS Example)
FAST2SMS_API_KEY = os.getenv("FAST2SMS_API_KEY") # Get from .env file
FAST2SMS_SENDER_ID = os.getenv("FAST2SMS_SENDER_ID", "FSTSMS") # Default or get from .env

# Email Settings (Gmail Example)
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")     # Your Gmail address (sender)
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")   # Your App Password (NOT your regular password)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

# Storage for annotated images
ANNOTATED_IMAGES_DIR = "annotated_images"
os.makedirs(ANNOTATED_IMAGES_DIR, exist_ok=True)

# --- Global State & Helper Classes ---
# Store active alerts and their state
active_alerts = {} # {alert_id: {"timestamp": datetime, "contact_number": str, "contact_email": str, "cancelled": bool, "detection_count": int, "image_path": str}}
MAX_ALERT_PENDING_TIME_SECONDS = 60 # Time window for user to cancel alert

# Location Info (UPDATE THIS for your specific location/demo)
LOCATION_NAME = "Chennai OMR Road, Near Sholinganallur"
LOCATION_LAT = 12.9126
LOCATION_LON = 80.2281

class AlertPayload(BaseModel):
    contact_number: str
    contact_email: str
    image_base64: Optional[str] = None # Not used in current FastAPI, but good for future extension

# --- FastAPI App Setup ---
app = FastAPI(
    title="ResQ: AI Road Accident Detection API",
    description="API for real-time accident detection, alerting, and human-in-the-loop cancellation.",
    version="1.0.0",
)

# Load YOLO model globally
try:
    model = YOLO(MODEL_PATH)
    print(f"INFO: YOLO model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"ERROR: Failed to load YOLO model from {MODEL_PATH}: {e}")
    # Consider raising an exception here to prevent the app from starting if model is essential
    raise RuntimeError(f"YOLO model failed to load. Check MODEL_PATH: {MODEL_PATH}")

# --- Helper Functions ---

def send_sms_alert(phone_number: str, message: str):
    if not FAST2SMS_API_KEY:
        print("WARNING: FAST2SMS_API_KEY not set. SMS alert skipped.")
        return False
    
    url = "https://www.fast2sms.com/devUtility/sms"
    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type": "application/json",
        "cache-control": "no-cache"
    }
    payload = json.dumps({
        "sender_id": FAST2SMS_SENDER_ID,
        "message": message,
        "language": "english",
        "route": "p", # Promotional route (usually free/cheaper for testing) - check Fast2SMS docs
        "numbers": phone_number
    })
    
    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        sms_response = response.json()
        if sms_response.get("return"): # Fast2SMS returns "true" for success
            print(f"INFO: SMS alert sent successfully to {phone_number}: {sms_response}")
            return True
        else:
            print(f"ERROR: Failed to send SMS to {phone_number}: {sms_response}")
            return False
    except requests.exceptions.Timeout:
        print(f"ERROR: SMS request to {phone_number} timed out.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: SMS sending failed for {phone_number}: {e}")
    return False


def send_email_alert(recipient_email: str, subject: str, body: str):
    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("WARNING: EMAIL_ADDRESS or EMAIL_PASSWORD not set. Email alert skipped.")
        return False

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = recipient_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls() # Secure the connection
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"INFO: Email alert sent successfully to {recipient_email}")
        return True
    except smtplib.SMTPAuthenticationError:
        print(f"ERROR: Email authentication failed. Check EMAIL_ADDRESS and EMAIL_PASSWORD (App Password for Gmail).")
    except smtplib.SMTPConnectError:
        print(f"ERROR: Could not connect to SMTP server {SMTP_SERVER}:{SMTP_PORT}. Check server address and port.")
    except Exception as e:
        print(f"ERROR: Failed to send email to {recipient_email}: {e}")
    return False

def process_detection(image: np.ndarray, contact_number: str, contact_email: str):
    current_time = datetime.now()
    frame_id = current_time.strftime("%Y%m%d%H%M%S%f") # Unique ID for frame

    results = model(image) # Perform inference with YOLOv8

    crash_detected_in_frame = False
    annotated_image_filename = None
    severity = "none" # Default severity

    for r in results:
        # Check if 'crashed_vehicle' (class 0 based on typical training) is in the results
        # Assuming 'crashed_vehicle' is class_id = 0 as per common YOLO datasets / custom training
        # You might need to adjust class_id based on your actual training
        detected_class_ids = [int(box.cls) for box in r.boxes]
        
        # Check for multiple detections or specific class ID
        if 0 in detected_class_ids: # Assuming 'crashed_vehicle' is class 0
            # Filter by confidence for 'crashed_vehicle'
            filtered_boxes = [
                box for box in r.boxes if int(box.cls) == 0 and float(box.conf) > CRASH_CONF_THRESHOLD
            ]
            if len(filtered_boxes) > 0:
                crash_detected_in_frame = True
                
                # Determine severity based on number of detected crash objects or their size/overlap
                # This is a simple example, refine as needed for your model
                if len(filtered_boxes) >= 2:
                    severity = "severe" # Multiple crash objects
                elif len(filtered_boxes) == 1:
                    severity = "moderate" # Single crash object
                else:
                    severity = "minor" # Less confident detection, or very small object

                # Draw bounding boxes and save annotated image
                annotated_img = r.plot() # YOLOv8's built-in plot method
                annotated_image_filename = f"crash_detection_{frame_id}.jpg"
                annotated_image_path = os.path.join(ANNOTATED_IMAGES_DIR, annotated_image_filename)
                cv2.imwrite(annotated_image_path, annotated_img)
                print(f"INFO: Annotated image saved to {annotated_image_path}")
                break # Only need one detection to flag as crash

    # --- Alert Logic ---
    if crash_detected_in_frame:
        # Check if an active alert already exists that hasn't been cancelled
        active_alert_id = None
        for alert_id, alert_data in active_alerts.items():
            if not alert_data["cancelled"] and (datetime.now() - alert_data["timestamp"]).total_seconds() < MAX_ALERT_PENDING_TIME_SECONDS:
                active_alert_id = alert_id
                break

        if active_alert_id:
            # Increment detection count for existing active alert
            active_alerts[active_alert_id]["detection_count"] += 1
            print(f"INFO: Existing alert {active_alert_id} - Consecutive detections: {active_alerts[active_alert_id]['detection_count']}")
            if active_alerts[active_alert_id]["detection_count"] >= MIN_CRASH_DETECTIONS_REQUIRED and not active_alerts[active_alert_id].get("alert_sent"):
                # Send actual alerts only after consecutive detections
                threading.Thread(target=trigger_emergency_alerts, args=(active_alert_id, active_alerts[active_alert_id])).start()
                active_alerts[active_alert_id]["alert_sent"] = True # Mark as sent
        else:
            # New potential alert, create a new entry
            new_alert_id = str(uuid.uuid4())
            active_alerts[new_alert_id] = {
                "timestamp": current_time,
                "contact_number": contact_number,
                "contact_email": contact_email,
                "cancelled": False,
                "detection_count": 1,
                "image_path": annotated_image_filename, # Store filename only
                "severity": severity,
                "alert_sent": False # To track if alerts have been dispatched
            }
            print(f"INFO: New potential alert created: {new_alert_id}")
            # If MIN_CRASH_DETECTIONS_REQUIRED is 1, trigger immediately
            if MIN_CRASH_DETECTIONS_REQUIRED == 1:
                threading.Thread(target=trigger_emergency_alerts, args=(new_alert_id, active_alerts[new_alert_id])).start()
                active_alerts[new_alert_id]["alert_sent"] = True

        return {
            "crash_detected": True,
            "alert_id": active_alert_id if active_alert_id else new_alert_id,
            "message": "Crash detected! Alert pending.",
            "image_url": f"/{ANNOTATED_IMAGES_DIR}/{annotated_image_filename}" if annotated_image_filename else None,
            "severity": severity
        }
    else:
        # No crash detected, reset consecutive count for all active alerts (if any)
        # Or, invalidate old alerts
        alerts_to_remove = []
        for alert_id, alert_data in active_alerts.items():
            if not alert_data["cancelled"] and (datetime.now() - alert_data["timestamp"]).total_seconds() >= MAX_ALERT_PENDING_TIME_SECONDS:
                alerts_to_remove.append(alert_id)
            # Optionally, reset detection count if no crash in current frame
            # alert_data["detection_count"] = 0 # This might make it too sensitive to single missed frames
        
        for alert_id in alerts_to_remove:
            print(f"INFO: Alert {alert_id} expired without cancellation or dispatch.")
            del active_alerts[alert_id]

        return {
            "crash_detected": False,
            "message": "No crash detected.",
            "alert_id": None,
            "image_url": None,
            "severity": "none"
        }

def trigger_emergency_alerts(alert_id: str, alert_data: dict):
    # This function runs in a separate thread
    contact_number = alert_data["contact_number"]
    contact_email = alert_data["contact_email"]
    severity = alert_data["severity"].upper()
    image_filename = alert_data["image_path"]

    sms_message = f"EMERGENCY: Road Accident Detected! Severity: {severity}. Location: {LOCATION_NAME} ({LOCATION_LAT}, {LOCATION_LON}). View image: {alert_id} - CHECK DASHBOARD." # Add dynamic URL later
    email_subject = f"ResQ ALERT: Critical Road Accident Detected - ID {alert_id} (Severity: {severity})"
    email_body = f"""
    Dear Emergency Contact,

    An automated accident detection system (ResQ) has identified a potential road accident.

    **Alert ID:** {alert_id}
    **Severity:** {severity}
    **Location:** {LOCATION_NAME}
                 Latitude: {LOCATION_LAT}, Longitude: {LOCATION_LON}
    **Timestamp:** {alert_data["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}

    Please check the ResQ dashboard for the annotated image and more details.
    (If this is a false alarm, please cancel the alert on the dashboard within {MAX_ALERT_PENDING_TIME_SECONDS} seconds.)

    ---
    This is an automated message. Do not reply.
    """
    
    print(f"\n--- TRIGGERING ALERTS for {alert_id} ---")
    sms_sent = send_sms_alert(contact_number, sms_message)
    email_sent = send_email_alert(contact_email, email_subject, email_body)

    if sms_sent or email_sent:
        print(f"INFO: Emergency alerts dispatched for {alert_id}.")
        # Schedule a check to see if alert was cancelled after pending time
        threading.Timer(MAX_ALERT_PENDING_TIME_SECONDS, check_alert_cancellation, args=[alert_id]).start()
    else:
        print(f"ERROR: No alerts were successfully sent for {alert_id}.")

def check_alert_cancellation(alert_id: str):
    if alert_id in active_alerts:
        if not active_alerts[alert_id]["cancelled"]:
            print(f"INFO: Alert {alert_id} was NOT cancelled by user within {MAX_ALERT_PENDING_TIME_SECONDS} seconds. Assuming confirmed incident.")
            # Here you might add logic for escalated alerts, logging to a database, etc.
        else:
            print(f"INFO: Alert {alert_id} was successfully cancelled by user.")
        
        # Clean up expired/handled alert from active_alerts dictionary
        # del active_alerts[alert_id] # Be careful with deletion if other threads might still access it
    else:
        print(f"WARNING: Alert {alert_id} not found in active_alerts during cancellation check.")

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "ResQ AI Accident Detection API is running!"}

@app.post("/detect_and_alert/")
async def detect_and_alert(
    file: UploadFile = File(...),
    contact_number: str = Form(...),
    contact_email: str = Form(...)
):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")
        
        # Process detection and alert logic
        response_data = process_detection(image, contact_number, contact_email)
        
        return JSONResponse(content=response_data)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unhandled error in /detect_and_alert/: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/annotated_images/{image_name}")
async def get_annotated_image(image_name: str):
    image_path = os.path.join(ANNOTATED_IMAGES_DIR, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(image_path, media_type="image/jpeg")

@app.post("/cancel_alert/{alert_id}")
async def cancel_alert(alert_id: str):
    if alert_id in active_alerts:
        if not active_alerts[alert_id]["cancelled"]:
            active_alerts[alert_id]["cancelled"] = True
            print(f"INFO: Alert {alert_id} has been successfully marked as CANCELLED by user.")
            return {"message": f"Alert {alert_id} cancelled successfully."}
        else:
            return {"message": f"Alert {alert_id} was already cancelled.", "status": "already_cancelled"}
    else:
        raise HTTPException(status_code=404, detail=f"Alert ID {alert_id} not found or has expired.")

@app.get("/alert_status/{alert_id}")
async def get_alert_status(alert_id: str):
    alert_data = active_alerts.get(alert_id)
    if alert_data:
        # Provide a simplified status for frontend polling
        return {
            "alert_id": alert_id,
            "active": not alert_data["cancelled"] and (datetime.now() - alert_data["timestamp"]).total_seconds() < MAX_ALERT_PENDING_TIME_SECONDS,
            "cancelled": alert_data["cancelled"],
            "alert_sent": alert_data.get("alert_sent", False),
            "severity": alert_data["severity"],
            "detection_count": alert_data["detection_count"],
            "image_url": f"/{ANNOTATED_IMAGES_DIR}/{alert_data['image_path']}" if alert_data["image_path"] else None,
            "timestamp": alert_data["timestamp"].isoformat(),
            "location_name": LOCATION_NAME,
            "location_lat": LOCATION_LAT,
            "location_lon": LOCATION_LON
        }
    else:
        raise HTTPException(status_code=404, detail=f"Alert ID {alert_id} not found or expired.")

@app.get("/active_alerts/")
async def get_all_active_alerts():
    """
    Returns a list of all currently active (not cancelled, not expired) alerts.
    """
    current_active = {}
    for alert_id, alert_data in active_alerts.items():
        if not alert_data["cancelled"] and (datetime.now() - alert_data["timestamp"]).total_seconds() < MAX_ALERT_PENDING_TIME_SECONDS:
            current_active[alert_id] = {
                "alert_id": alert_id,
                "cancelled": alert_data["cancelled"],
                "alert_sent": alert_data.get("alert_sent", False),
                "severity": alert_data["severity"],
                "timestamp": alert_data["timestamp"].isoformat(),
                "image_url": f"/{ANNOTATED_IMAGES_DIR}/{alert_data['image_path']}" if alert_data["image_path"] else None,
                "location_name": LOCATION_NAME,
                "location_lat": LOCATION_LAT,
                "location_lon": LOCATION_LON
            }
    return JSONResponse(content=current_active)