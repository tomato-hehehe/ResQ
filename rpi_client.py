# rpi_client.py
import cv2
import requests
import time
import numpy as np
import base64 # Not used for current implementation, but good for base64 sending option

# --- Configuration (YOU MUST REPLACE THESE) ---
# 1. Phone Camera URL (from IP Webcam app)
# Find this in your IP Webcam app on your phone (e.g., "http://192.168.1.150:8080/video")
PHONE_CAM_URL = "http://YOUR_PHONE_IP:8080/video" # <--- REPLACE YOUR_PHONE_IP!

# 2. FastAPI Endpoint URL (where your app.py is running)
# During local testing: "http://YOUR_LAPTOP_IP:8000/detect_and_alert/"
# After cloud deployment: "https://YOUR_RENDER_URL.onrender.com/detect_and_alert/"
FASTAPI_ENDPOINT_URL = "http://YOUR_LAPTOP_IP:8000/detect_and_alert/" # <--- REPLACE THIS LATER!

# 3. Default Contact Information (sent with each frame)
DEFAULT_CONTACT_NUMBER = "9876543210" # <--- REPLACE with your real emergency contact number
DEFAULT_CONTACT_EMAIL = "your_emergency_email@example.com" # <--- REPLACE with your real emergency contact email

# 4. Frame Sending Rate
FRAME_SEND_INTERVAL_SECONDS = 5 # Send a frame to FastAPI every 5 seconds

# --- Script Logic ---

def send_frame_to_api(frame_bytes, contact_number, contact_email):
    files = {'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
    data = {
        'contact_number': contact_number,
        'contact_email': contact_email
    }
    
    try:
        response = requests.post(FASTAPI_ENDPOINT_URL, files=files, data=data, timeout=15) # Increased timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        api_response = response.json()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] API Response: {api_response}")
        
        if api_response.get("crash_detected"):
            alert_id = api_response.get("alert_id")
            severity = api_response.get("severity", "N/A").upper()
            print(f"*** CRASH DETECTED! Alert ID: {alert_id}, Severity: {severity} ***")
            # You might want to blink an LED on RPi, play a sound, etc. here

    except requests.exceptions.ConnectionError:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: Could not connect to FastAPI server at {FASTAPI_ENDPOINT_URL}. Is it running?")
    except requests.exceptions.Timeout:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: Request to FastAPI server timed out after 15 seconds.")
    except requests.exceptions.RequestException as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error sending frame to API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"API Error Response: {e.response.text}")
    except ValueError as e: # Catch JSONDecodeError specifically for invalid JSON
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error decoding JSON response from API: {e}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] An unexpected error occurred: {e}")

def capture_and_send():
    print(f"Attempting to connect to camera at: {PHONE_CAM_URL}")
    cap = cv2.VideoCapture(PHONE_CAM_URL)

    if not cap.isOpened():
        print(f"Error: Could not open video stream from {PHONE_CAM_URL}. Check camera URL or if phone is streaming.")
        return

    print("Camera connected. Starting frame capture and sending...")
    last_send_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame. Reconnecting to camera...")
                cap.release()
                cap = cv2.VideoCapture(PHONE_CAM_URL)
                if not cap.isOpened():
                    print("Could not reconnect. Waiting 5 seconds before retrying...")
                    time.sleep(5)
                continue

            current_time = time.time()
            if current_time - last_send_time >= FRAME_SEND_INTERVAL_SECONDS:
                # Encode frame to JPEG bytes
                _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                frame_bytes = img_encoded.tobytes()

                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sending frame to FastAPI...")
                send_frame_to_api(frame_bytes, DEFAULT_CONTACT_NUMBER, DEFAULT_CONTACT_EMAIL)
                last_send_time = current_time

            # Optional: Display frame locally (for debugging on RPi with monitor)
            # cv2.imshow('RPi Camera Feed', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # Small delay to prevent busy-waiting if frame processing is very fast
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting frame capture.")
    finally:
        cap.release()
        # cv2.destroyAllWindows() # If imshow was used

if __name__ == "__main__":
    capture_and_send()