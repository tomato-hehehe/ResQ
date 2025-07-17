# local_test_client.py
import requests
import os
import time
from datetime import datetime
import json # Import json for pretty printing

# --- Configuration (YOU MUST REPLACE THESE) ---
# IMPORTANT: This is the URL where YOUR FastAPI backend is running on YOUR laptop.
# If you run FastAPI with `uvicorn app:app --reload --host 0.0.0.0 --port 8000`,
# then YOUR_LAPTOP_IP will be your computer's actual local IP address.
# YOU WILL FIND THIS BELOW IN THE INSTRUCTIONS.
FASTAPI_ENDPOINT_URL = "http://YOUR_LAPTOP_IP:8000/detect_and_alert/" # <--- REPLACE YOUR_LAPTOP_IP!

# Path to a test image on your laptop
TEST_IMAGE_PATH = "testing1.jpg" # Make sure this image exists in the same directory as this script.
                                 # Or provide a full path: "C:/Users/YourUser/Projects/ResQ/testing1.jpg"

# Default contact info for alerts (these will be sent to the FastAPI backend)
DEFAULT_CONTACT_NUMBER = "9876543210" # <--- REPLACE with a real phone number for testing (e.g., your own)
DEFAULT_CONTACT_EMAIL = "your_emergency_email@example.com" # <--- REPLACE with a real email for testing (e.g., your own)

# --- Script Logic ---
if not os.path.exists(TEST_IMAGE_PATH):
    print(f"Error: Test image not found at {TEST_IMAGE_PATH}. Please provide a valid path to an image file.")
    exit()

print(f"Attempting to send image '{TEST_IMAGE_PATH}' to FastAPI at {FASTAPI_ENDPOINT_URL}")

# Read the image file as bytes
with open(TEST_IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

# Prepare the files and data for the POST request
# FastAPI expects 'file' as the UploadFile, and 'contact_number', 'contact_email' as Forms
files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
data = {
    'contact_number': DEFAULT_CONTACT_NUMBER,
    'contact_email': DEFAULT_CONTACT_EMAIL
}

try:
    response = requests.post(FASTAPI_ENDPOINT_URL, files=files, data=data, timeout=20) # Added timeout
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    api_response = response.json()

    print("\n--- API Response ---")
    print(json.dumps(api_response, indent=2)) # Pretty print JSON

    if api_response.get("crash_detected"):
        alert_id = api_response.get("alert_id", "N/A")
        severity = api_response.get("severity", "N/A")
        image_url_suffix = api_response.get('image_url', '')
        
        # Construct full public URL for the annotated image (for easy viewing)
        # This assumes your FastAPI server is accessible from where you're viewing.
        base_api_url = FASTAPI_ENDPOINT_URL.split('/detect_and_alert')[0]
        full_image_url = f"{base_api_url}{image_url_suffix}" if image_url_suffix else "No image URL provided."

        print(f"\n*** CRASH DETECTED! Alert ID: {alert_id}, Severity: {severity.upper()} ***")
        print(f"View Annotated Image: {full_image_url}\n")
    else:
        print("\nNo crash detected in the test image.")

except requests.exceptions.ConnectionError:
    print(f"Error: Could not connect to FastAPI server at {FASTAPI_ENDPOINT_URL}.")
    print("Please ensure your FastAPI (app.py) is running and accessible (check firewall).")
except requests.exceptions.Timeout:
    print("Error: Request to FastAPI server timed out after 20 seconds.")
except requests.exceptions.RequestException as e:
    print(f"Error sending frame to API: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"API Error Response: {e.response.text}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON response from API. Response was: {response.text}")

print("\nLocal test client finished.")