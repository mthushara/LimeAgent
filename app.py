#!/usr/bin/env python3
"""
NIC VERIFICATION SYSTEM - SINGLE-FILE WEB APP

Everything the old two-file setup needed (nic_verification_merged.py's core
engine + app.py's Flask front-end) lives in this one file. There is no
Tkinter GUI here and nothing runs a local desktop window or reads the
server's own webcam - this is meant to run headless on a cloud host, with
the verification photo either uploaded or captured via the BROWSER's camera
(getUserMedia), then POSTed to /api/verify.

Run with (dev only): python app.py
Offline signature check (no web server, no LIME):
    python NIC-Sig-NoENV-NF.py --compare reference.png captured1.png [...]
For real deployment use gunicorn - see DEPLOY.md.

This app surfaces two separate face-detection subsystems:
  - FaceRecognitionAgent  - the AI agent (insightface / face_recognition)
                            that scores how similar two face photos are.
  - LiveFaceDetector      - a fast mediapipe/YuNet/Haar detector, kept for
                            parity with the old capture-quality checks (not
                            used to read a live camera stream server-side).
Their status is exposed at GET /api/status.
"""

# =============================================================================
# CORE VERIFICATION ENGINE (from nic_verification_merged.py, GUI removed)
# =============================================================================

import os
import sys
import json
import time
import base64
import hashlib
import logging
import requests
import re
import threading
import queue
import glob
import difflib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
# Silence OpenCV's internal C++-level logging (e.g. the DNN "NaryEltwiseLayer
# shape mismatch" warnings YuNet prints when fed very small crops) - must be
# set before cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2
import numpy as np
from dotenv import load_dotenv
from skimage.metrics import structural_similarity as ssim
from scipy import ndimage
from scipy.optimize import minimize
from skimage.morphology import skeletonize
from PIL import Image
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# AI FACE RECOGNITION IMPORTS
# =============================================================================

AI_FACE_AVAILABLE = False
FACE_RECOGNITION_AVAILABLE = False
INSIGHTFACE_AVAILABLE = False

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    pass

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    pass

if FACE_RECOGNITION_AVAILABLE or INSIGHTFACE_AVAILABLE:
    AI_FACE_AVAILABLE = True

# -----------------------------------------------------------------------------
# LIVE FACE DETECTOR (used only during capture, to confirm a face is actually
# visible before a frame is accepted - separate from the matching models above)
# MediaPipe is tried first: it installs as a plain Python wheel (no C++ build
# tools required, unlike dlib/face_recognition) and is fast + accurate enough
# for real-time guidance. See LiveFaceDetector class for the full fallback chain.
# -----------------------------------------------------------------------------
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# CONFIGURATION
# =============================================================================

# BASE_DIR is defined here (moved up from the Flask front-end section further
# below) so both env files can be loaded from an absolute path - a bare
# load_dotenv() only auto-finds a file literally named ".env" in the current
# working directory, which silently does nothing if you launch the app from
# somewhere other than this folder (e.g. a scheduled task or another cwd).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PRD/LIME credentials (PRD_CHANNEL_KEY, LIME_AUTH_TOKEN) live in
# nic-verification.env. This is intentionally separate from webapp.env
# (login/session config), loaded later below.
#
# Resolution order for the credentials env file:
#   1. NIC_CREDENTIALS_ENV env var, if set (absolute path - e.g. a mounted
#      secret in the container).
#   2. D:\ENV\nic-verification.env (fixed local-dev location).
# In DTE / any container, the real values are normally injected as
# environment variables directly, in which case load_dotenv is a harmless
# no-op (it never overrides variables already present in the environment).
#=>_CREDENTIALS_ENV = os.environ.get("NIC_CREDENTIALS_ENV") or r"D:\ENV\nic-verification.env"
#=>if os.path.isfile(_CREDENTIALS_ENV):
#=>    load_dotenv(_CREDENTIALS_ENV)

# Load .env file
load_dotenv()

logging.basicConfig(level=logging.WARNING)

PRD_CHANNEL_KEY = os.getenv("PRD_CHANNEL_KEY")
#==> IMAGE_FOLDER = os.getenv("IMAGE_FOLDER", r"D:\Test-Ai\Signature_Base")

# API Endpoints
PRD_BASE_URL = "https://omcloud.dialog.lk/om-drp-gateway"
PRD_QUERY_URL_TEMPLATE = f"{PRD_BASE_URL}/nic/validateNicByDrp/{{nic}}"

# Additional Configurations
LIME_AUTH_TOKEN = os.getenv("LIME_AUTH_TOKEN")
NIC_STORAGE_FOLDER = os.getenv("NIC_STORAGE_FOLDER", r"D:\Test-Ai\NIC")

# Base URLs
LIME_BASE_URL = "https://limedms.dialog.lk/limedms-apigw"

# API Endpoints
LIME_SEARCH_URL = f"{LIME_BASE_URL}/document/search?NIC_PP_BRC={{}}"
LIME_DOWNLOAD_URL = f"{LIME_BASE_URL}/download/{{}}"

# DocType-scoped search - LIME's own portal ("Document View By Any Field")
# has been observed finding KYC/NIC documents that the generic
# document/search above misses (e.g. docs attached to a SIM-change or other
# transaction). Exact parameter names are unconfirmed (no Swagger access),
# so this follows the same AccountNo query-param convention with DocType
# added - see search_lime_documents_by_doctype for how a wrong guess here
# is surfaced instead of silently failing.
LIME_SEARCH_BY_DOCTYPE_URL = f"{LIME_BASE_URL}/document/search/docType"

# NIC Card dimensions
NIC_CARD_WIDTH_MM = 54.0
NIC_CARD_HEIGHT_MM = 86.0
NIC_ASPECT_RATIO = NIC_CARD_WIDTH_MM / NIC_CARD_HEIGHT_MM

# Auto-capture settings - INCREASED TO 10 SECONDS
AUTO_CAPTURE_DELAY = 10  # seconds - increased for proper positioning
CAPTURE_GRACE_EXTENSION = 5  # extra seconds allowed if no face has been found yet
CAPTURE_QUALITY_THRESHOLD = 0.3  # Lowered threshold for better capture
FACE_MATCH_THRESHOLD = 0.6
SSIM_THRESHOLD = 0.70
MIN_FACE_CONFIDENCE = 0.5

# Signature comparison thresholds for the LOCAL signature matcher (0-100
# score from compare_signature_images_combined - ink alignment + stroke
# closeness/coverage/shape context; see the SIGNATURE VERIFICATION section).
# Calibrated on a genuine pair (scanned specimen vs phone photo, ~62) and
# synthetic impostors (mostly <47, hardest look-alike ~56). Signature
# verification never uses the face agent or compare_images_combined().
# Override in nic-verification.env (this is read before webapp.env loads)
# once calibrated on real LIME data with:
#   python NIC-Sig-NoENV-NF.py --compare reference.png captured.png
SIGNATURE_APPROVE_THRESHOLD = float(os.getenv("SIGNATURE_APPROVE_THRESHOLD", "57"))
SIGNATURE_REVIEW_THRESHOLD = float(os.getenv("SIGNATURE_REVIEW_THRESHOLD", "45"))

# -----------------------------------------------------------------------------
# Base64.ai Signature AI - used ONLY by the signature verification flow
# (compare_signatures() / verify_signature_with_base64ai() below). This does
# NOT touch NIC/face verification in any way - PRD and the local
# FaceRecognitionAgent/LiveFaceDetector are untouched and keep working exactly
# as before, even if Base64.ai is unreachable or these settings are left blank.
#
# Auth per Base64.ai's API docs (https://apidoc.base64.ai): the Authorization
# header must be "ApiKey <email>:<secret>", where <email> is the Base64.ai
# account login email and <secret> is the API Key from the Integrations page
# (https://base64.ai/integrations).
# -----------------------------------------------------------------------------
BASE64AI_EMAIL = os.getenv("BASE64AI_EMAIL", "")
BASE64AI_API_KEY = os.getenv("BASE64AI_API_KEY", "")
BASE64AI_SIGNATURE_URL = os.getenv("BASE64AI_SIGNATURE_URL", "https://base64.ai/api/signature")
BASE64AI_ENABLED = bool(BASE64AI_EMAIL and BASE64AI_API_KEY)
# Max number of signatures Base64.ai should look for per image, and its own
# detection-confidence threshold - see section 4.1 of the Base64.ai API docs.
BASE64AI_MAX_SIGNATURES = int(os.getenv("BASE64AI_MAX_SIGNATURES", "1"))
BASE64AI_DETECTION_THRESHOLD = float(os.getenv("BASE64AI_SIGNATURE_DETECTION_THRESHOLD", "0.4"))
# Request timeout - Base64.ai's docs quote a P99 response time under 3s for
# /api/signature, so 20s leaves comfortable headroom for a slow network hop.
BASE64AI_TIMEOUT_SECONDS = 20

# Quality requirements - LOWERED for better capture success
MIN_QUALITY_SCORE = 0.20  # Lowered from 0.50
MIN_SHARPNESS = 30  # Lowered from 100
MIN_EDGE_DENSITY = 0.01  # Lowered from 0.02

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']

# LIME Document Types for NIC
LIME_DOC_TYPES = ['KYC_FRONT', 'NIC_FRONT', 'KYC_Front', 'NIC_Front', 'KYC', 'NIC']

# LIME Document Types for the customer's stored SIGNATURE document. Kept as
# a separate alias list (same pattern as LIME_DOC_TYPES above) since LIME
# environments/doc templates have been observed to label this doc type
# slightly differently (e.g. plain "SIGNATURE" vs "CUSTOMER_SIGNATURE").
# The matching in extract_signature_documents_from_lime() is a substring
# check against the normalized DocType (uppercased, "_"/"-" -> space), so
# e.g. "SIGNATURE_CARD" or "SIGNATURE_SPECIMEN" already match the plain
# "SIGNATURE" alias below without needing their own separate entry - the
# extra aliases here only cover cases where "SIGNATURE" itself doesn't
# appear as a substring at all (e.g. abbreviated to "SIG"/"SIGN").
LIME_SIGNATURE_DOC_TYPES = [
    'SIGNATURE', 'CUSTOMER_SIGNATURE', 'CUST_SIGNATURE', 'SIGNATURE_CARD',
    'SIGNATURE_SPECIMEN', 'SPECIMEN_SIGNATURE', 'SIGN', 'SIGN_CARD',
    'CUSTOMER_SIGN', 'SIG', 'SIG_CARD',
]

# Recent images file
RECENT_IMAGES_FILE = "recent_images.json"

# Dedicated audit-trail file for user search/verification activity - kept
# separate from verification_logs.txt (which is full of general debug/
# technical noise: face-detector startup messages, LIME warnings, raw
# metadata dumps, etc.) so the actual "who searched what, and what was the
# result" trail is easy to read on its own.
VERIFICATION_SEARCH_LOG_FILE = os.path.join(BASE_DIR, "Verification_search_log.txt")

# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ImageComparisonError(Exception):
    pass

class PRDAPIError(Exception):
    pass

class LocalImageNotFoundError(Exception):
    pass

class NoDocumentsFoundException(Exception):
    pass

class NoCustomerFoundException(Exception):
    pass

class FaceDetectionError(Exception):
    pass

# =============================================================================
# NIC VALIDATION FUNCTIONS
# =============================================================================

def validate_nic(cust_nic):
    """Validate and convert NIC between old and new formats"""
    old_nic = ""
    new_nic = ""
    is_diff = "Y"

    cust_nic = str(cust_nic).strip().upper()

    if len(cust_nic) == 12:
        if cust_nic[0] == "1":
            if cust_nic[7] == "1":
                old_nic = cust_nic
                new_nic = cust_nic
                is_diff = "N"
            else:
                old_nic = cust_nic[2:7] + cust_nic[8:12] + "V"
                new_nic = cust_nic
        else:
            old_nic = cust_nic
            new_nic = cust_nic
            is_diff = "N"
    else:
        if cust_nic[0].isdigit():
            clean_nic = cust_nic.replace('V', '').replace('v', '')
            if len(clean_nic) == 9:
                new_nic = "19" + clean_nic[0:5] + "0" + clean_nic[5:9]
            else:
                new_nic = cust_nic
        else:
            new_nic = cust_nic
        
        old_nic = cust_nic
        
        if old_nic == new_nic:
            is_diff = "N"

    return {
        "oldNIC": old_nic,
        "newNIC": new_nic,
        "isDiff": is_diff
    }

def normalize_nic(nic):
    """Convert NIC to standard format"""
    if not nic:
        return ""
    
    nic = str(nic).strip().upper()
    validation_result = validate_nic(nic)
    
    if validation_result["newNIC"] and validation_result["isDiff"] == "Y":
        return validation_result["newNIC"]
    else:
        if nic.startswith('N'):
            nic = nic[1:]
        if not nic.endswith('V') and len(nic) == 9:
            nic = nic + 'V'
        return nic

def normalize_nic_old(nic):
    """Convert NIC to old format"""
    if not nic:
        return ""
    
    nic = str(nic).strip().upper()
    validation_result = validate_nic(nic)
    
    if validation_result["oldNIC"]:
        return validation_result["oldNIC"]
    else:
        if nic.startswith('N'):
            nic = nic[1:]
        if not nic.endswith('V') and len(nic) == 9:
            nic = nic + 'V'
        return nic

def are_nics_equivalent(nic1, nic2):
    """Check if two NICs represent the same identity"""
    n1 = normalize_nic(nic1) if nic1 else ""
    n2 = normalize_nic(nic2) if nic2 else ""
    
    if not n1 or not n2:
        return False
    
    if n1 == n2:
        return True
    
    old1 = normalize_nic_old(nic1)
    old2 = normalize_nic_old(nic2)
    
    if old1 == old2:
        return True
    
    clean1 = n1.replace('V', '').replace('v', '')
    clean2 = n2.replace('V', '').replace('v', '')
    
    if clean1 == clean2:
        return True
    
    if len(clean1) == 12 and len(clean2) < 12:
        return clean1[4:] == clean2
    if len(clean2) == 12 and len(clean1) < 12:
        return clean2[4:] == clean1
    
    return False

def clean_nic_for_api(nic):
    """Clean NIC for API call"""
    if not nic:
        return ""
    
    try:
        validation = validate_nic(nic)
        if validation["newNIC"] and validation["isDiff"] == "Y":
            return validation["newNIC"]
        elif validation["oldNIC"]:
            return validation["oldNIC"]
    except:
        pass
    
    return str(nic).strip()

# PRD only accepts NIC in exactly one of these two shapes. Checked locally
# before every PRD call so a bad/garbage NIC (wrong field, stray input,
# missing LIME metadata, etc.) fails fast with a clear message instead of
# reaching PRD and coming back as a confusing gateway 400.
NIC_FORMAT_PATTERN = re.compile(r'^(\d{9}[Vv]|\d{12})$')

def is_valid_nic_format(nic):
    """True if nic matches PRD's required shape: 9 digits+V/v or 12 digits"""
    return bool(nic) and bool(NIC_FORMAT_PATTERN.match(str(nic).strip()))

def resolve_nic_for_prd(lime_nic, nic_number):
    """Pick the NIC to send to PRD (preferring LIME's), clean it the same
    way the working stored_prd path does, and validate its format before
    ever calling PRD. Returns (clean_nic, source, error_message).
    source is 'lime' or 'manual', for logging/debugging which one was used.
    """
    if lime_nic:
        source = "lime"
        raw = lime_nic
    elif nic_number:
        source = "manual"
        raw = nic_number
    else:
        return None, None, (
            "Could not determine NIC - not found via LIME lookup and none "
            "was provided. A mobile number alone cannot be used as a NIC."
        )

    clean = clean_nic_for_api(raw)
    if not is_valid_nic_format(clean):
        return None, source, (
            f"NIC from {source} ('{raw}') is not a valid NIC "
            f"(must be 9 digits followed by V/v, or 12 digits)."
        )
    log_message(f"Using NIC for PRD query: source={source} value={clean}", "INFO")
    return clean, source, None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_mobile_number(mobile):
    """Validate Sri Lankan mobile number format"""
    patterns = [r'^0[0-9]{9}$', r'^[0-9]{10}$', r'^\+94[0-9]{9}$']
    for pattern in patterns:
        if re.match(pattern, mobile):
            return True
    return False

def format_mobile_number(mobile):
    """Format mobile number to standard format"""
    mobile = re.sub(r'[^0-9]', '', mobile)
    if mobile.startswith('94'):
        mobile = '0' + mobile[2:]
    if not mobile.startswith('0'):
        mobile = '0' + mobile
    return mobile

def log_message(message, level="INFO"):
    """Log messages to file"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("verification_logs.txt", "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] [{level}] {message}\n")
    except:
        pass

def save_verification_search_log(username, nic="", mobile="", result_status=""):
    """Append a single, lightweight line to VERIFICATION_SEARCH_LOG_FILE for
    every search/verify request - just who ran it, what was searched, and
    the outcome. Deliberately does NOT store the NIC image or any of the
    raw PRD/LIME payload (unlike the old verification_history.json, which
    grew very large because it kept the full result - including images -
    for every request).

    This is intentionally a SEPARATE file from verification_logs.txt (the
    general debug/technical log written by log_message()) so the user
    activity audit trail - login user, NIC/mobile searched, result, time -
    stays readable on its own instead of being buried among face-detector
    startup messages, LIME warnings, and raw metadata dumps."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nic_part = nic or "-"
        mobile_part = mobile or "-"
        status_part = (result_status or "UNKNOWN").upper()
        with open(VERIFICATION_SEARCH_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"[{timestamp}] [VERIFY] user={username} nic={nic_part} "
                f"mobile={mobile_part} result={status_part}\n"
            )
        return True
    except Exception:
        return False

# =============================================================================
# RECENT IMAGES MANAGER
# =============================================================================

class RecentImagesManager:
    """Manage recent images for quick access"""
    
    def __init__(self):
        self.recent_paths = self._load_recent()
        self.max_recent = 20
    
    def _load_recent(self):
        """Load recent images from file"""
        try:
            if os.path.exists(RECENT_IMAGES_FILE):
                with open(RECENT_IMAGES_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('images', [])
        except:
            pass
        return []
    
    def _save_recent(self):
        """Save recent images to file"""
        try:
            with open(RECENT_IMAGES_FILE, 'w', encoding='utf-8') as f:
                json.dump({'images': self.recent_paths}, f, indent=2)
        except:
            pass
    
    def add_image(self, path):
        """Add image to recent list"""
        if not os.path.exists(path):
            return
        
        if path in self.recent_paths:
            self.recent_paths.remove(path)
        
        self.recent_paths.insert(0, path)
        
        if len(self.recent_paths) > self.max_recent:
            self.recent_paths = self.recent_paths[:self.max_recent]
        
        self._save_recent()
    
    def get_recent(self):
        """Get list of recent images"""
        self.recent_paths = [p for p in self.recent_paths if os.path.exists(p)]
        self._save_recent()
        return self.recent_paths
    
    def get_stored_images(self, folder_path=None):
        """Get all stored images from folder"""
        if folder_path is None:
            folder_path = NIC_STORAGE_FOLDER
        
        images = []
        if os.path.exists(folder_path):
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                pattern = os.path.join(folder_path, f"*{ext}")
                for file_path in glob.glob(pattern):
                    images.append(file_path)
                pattern = os.path.join(folder_path, f"*{ext.upper()}")
                for file_path in glob.glob(pattern):
                    if file_path not in images:
                        images.append(file_path)
        
        images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return images

# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================

def decode_base64_image(base64_string):
    """Decode base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(BytesIO(image_bytes))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        opencv_image = np.array(pil_image)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        raise ImageComparisonError(f"Failed to decode base64 image: {str(e)}")

def encode_image_to_base64(image, ext=".jpg"):
    """Encode an OpenCV (BGR numpy) image to a plain base64 string - the
    reverse of decode_base64_image(). No 'data:image/...' prefix, matching
    the plain base64 body PRD's validateNicByDrp expects for nicFrontImage.
    """
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise ImageComparisonError("Failed to encode image for PRD upload")
    return base64.b64encode(buffer).decode("utf-8")

def encode_image_to_base64_data_url(image, ext=".jpg"):
    """Encode an OpenCV (BGR numpy) image to a 'data:image/...;base64,...'
    string - the format Base64.ai's /api/signature endpoint requires for its
    'document'/'query' fields (see encode_image_to_base64() above for the
    plain-base64 variant PRD uses instead)."""
    mime = "image/png" if ext.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encode_image_to_base64(image, ext=ext)}"

def load_local_image(image_path):
    """Load local image from file path"""
    if not os.path.exists(image_path):
        raise LocalImageNotFoundError(f"Image not found: {image_path}")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ImageComparisonError(f"Failed to load image: {image_path}")
        return image
    except Exception as e:
        raise ImageComparisonError(f"Error loading image: {str(e)}")

def preprocess_image(image):
    """Preprocess image for better comparison"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    return blurred

def resize_images(image1, image2, target_size=(512, 512)):
    """Resize images to same dimensions"""
    if len(image1.shape) == 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = image1
    
    if len(image2.shape) == 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = image2
    
    resized1 = cv2.resize(gray1, target_size, interpolation=cv2.INTER_AREA)
    resized2 = cv2.resize(gray2, target_size, interpolation=cv2.INTER_AREA)
    
    return resized1, resized2

def compare_ssim(image1, image2):
    """Compare images using SSIM"""
    try:
        processed1 = preprocess_image(image1)
        processed2 = preprocess_image(image2)
        resized1, resized2 = resize_images(processed1, processed2)
        
        similarity, _ = ssim(resized1, resized2, full=True)
        return similarity * 100
    except:
        return 0.0

def compare_histogram(image1, image2):
    """Compare images using histogram correlation"""
    try:
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1
            
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2
        
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, correlation * 100)
    except:
        return 0.0

def compare_mse(image1, image2):
    """Compare images using MSE"""
    try:
        processed1 = preprocess_image(image1)
        processed2 = preprocess_image(image2)
        resized1, resized2 = resize_images(processed1, processed2)
        
        mse = np.mean((resized1.astype(float) - resized2.astype(float)) ** 2)
        max_mse = 255 ** 2
        return max(0, (1 - (mse / max_mse)) * 100)
    except:
        return 0.0

def compare_orb(image1, image2):
    """Compare images using ORB feature matching"""
    try:
        orb = cv2.ORB_create()
        
        keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
        
        if descriptors1 is None or descriptors2 is None:
            return 0.0
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        num_matches = len(matches)
        if num_matches == 0:
            return 0.0
        
        top_matches = matches[:max(10, num_matches // 2)]
        good_matches = [m for m in top_matches if m.distance < 50]
        
        if num_matches > 0:
            return min(100, (len(good_matches) / min(num_matches, 200)) * 100)
        return 0.0
    except:
        return 0.0

def compare_images_combined(image1, image2):
    """Combined comparison using multiple methods"""
    results = {}
    
    try:
        results['ssim'] = compare_ssim(image1, image2)
    except:
        results['ssim'] = 0.0
    
    try:
        results['histogram'] = compare_histogram(image1, image2)
    except:
        results['histogram'] = 0.0
    
    try:
        results['mse'] = compare_mse(image1, image2)
    except:
        results['mse'] = 0.0
    
    try:
        results['orb'] = compare_orb(image1, image2)
    except:
        results['orb'] = 0.0
    
    weights = {'ssim': 0.40, 'histogram': 0.20, 'mse': 0.20, 'orb': 0.20}
    overall = sum(results[key] * weights[key] for key in weights.keys())
    
    return overall, results

def calculate_image_quality(image):
    """Calculate image quality score using Laplacian variance"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range with improved scaling
        score = min(1.0, laplacian_var / 200)
        return score
    except:
        return 0.3  # Default moderate quality

def detect_edges(image):
    """Detect edges in image for shape verification"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return edge_density
    except:
        return 0.0

def _find_best_card_contour(contours):
    """
    Given a list of contours, find the best card-shaped candidate using a
    minAreaRect fit rather than requiring an exact 4-point polygon match.
    This tolerates rounded card corners, slight rotation, and minor glare
    breaks in the outline - all of which commonly cause an exact
    approxPolyDP(...) == 4 check to miss a real card.

    Orientation-independent: works whether the card is held portrait or
    landscape, by comparing short_side/long_side rather than raw width/height.
    """
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 3000:
            continue

        rect = cv2.minAreaRect(contour)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0:
            continue

        long_side, short_side = max(rw, rh), min(rw, rh)
        ratio = short_side / long_side  # orientation-independent, always <= 1

        # Sri Lanka NIC card ratio is ~0.628 (54mm x 86mm) - accept a fairly
        # wide band since perspective/rotation skews the measured ratio a bit
        if not (0.45 < ratio < 0.80):
            continue

        box = cv2.boxPoints(rect).astype(np.int32)
        x, y, w, h = cv2.boundingRect(box)
        # OpenCV 5.0's Python bindings can return NumPy scalar types (e.g.
        # numpy.int32) here rather than plain Python ints. Casting now stops
        # that type from propagating into face-bbox math, size checks, and
        # drawing calls further downstream, where OpenCV 5.x + NumPy 2.x can
        # reject NumPy scalars with a confusing "wrong type" error.
        candidates.append((area, (int(x), int(y), int(w), int(h))))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def detect_nic_region(image):
    """
    Detect the NIC card region in an image.

    Tries two independent preprocessing paths and uses whichever finds a
    plausible card-shaped contour first, since a single adaptive-threshold
    pass alone regularly misses the card under glare, uneven lighting, or a
    busy/textured background:
      1. Adaptive threshold - works well on a plain, evenly-lit background
      2. Canny edges + morphological closing - more robust to glare on a
         laminated card and uneven lighting, where the adaptive-threshold
         outline often breaks up
    Accepts the card in EITHER portrait or landscape orientation.
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Path 1: adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        contours1, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = _find_best_card_contour(contours1)
        if result:
            return result

        # Path 2: Canny edges + morphological closing (more robust to glare/shadow)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours2, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return _find_best_card_contour(contours2)
    except:
        return None

def crop_nic_region(image):
    """Crop NIC region from image"""
    rect = detect_nic_region(image)
    if rect:
        x, y, w, h = rect
        padding = int(min(w, h) * 0.05)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        return image[y:y+h, x:x+w]
    return image

def check_nic_size_and_quality(image):
    """Check if image has proper NIC size and quality - orientation-independent, looser constraints"""
    rect = detect_nic_region(image)

    if rect is None:
        return {
            "valid": False,
            "reason": "NIC card not detected in frame",
            "score": 0
        }

    x, y, w, h = rect
    long_side, short_side = max(w, h), min(w, h)
    aspect_ratio = short_side / long_side if long_side else 0
    area = w * h
    image_area = image.shape[0] * image.shape[1]
    coverage = area / image_area

    # Orientation-independent aspect ratio check (card may be held portrait or landscape)
    if not (0.45 < aspect_ratio < 0.80):
        return {
            "valid": False,
            "reason": f"NIC aspect ratio: {aspect_ratio:.2f} (expected ~0.63, either orientation)",
            "score": 0.1
        }

    # Looser coverage check
    if coverage < 0.05:
        return {
            "valid": False,
            "reason": f"NIC too small: {coverage:.1%} of frame (minimum 5%)",
            "score": coverage
        }

    # Looser size check (based on the card's short side, orientation-independent)
    if short_side < 80:
        return {
            "valid": False,
            "reason": f"NIC too small: {int(short_side)}px on short side (minimum 80px)",
            "score": coverage
        }

    return {
        "valid": True,
        "reason": "Proper NIC size detected",
        "score": min(1.0, coverage * 3),
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio
    }

def detect_glare_and_brightness(image, region=None):
    """
    Detect overexposed glare spots (very common on laminated NIC cards under
    indoor lighting) and check overall brightness. This is a frequent, silent
    cause of poor face-photo capture: the frame looks 'sharp' and 'has a face'
    but the actual photo area is blown out or too dark to be useful.

    Returns: {brightness, glare_ratio, has_glare, too_dark, too_bright}
    """
    try:
        if region:
            x, y, w, h = region
            crop = image[max(0, y):y + h, max(0, x):x + w]
        else:
            crop = image

        if crop is None or crop.size == 0:
            crop = image

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        brightness = float(np.mean(gray))

        # Glare = clusters of near-white, blown-out pixels
        _, overexposed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        glare_ratio = float(np.sum(overexposed > 0)) / overexposed.size

        return {
            "brightness": brightness,
            "glare_ratio": glare_ratio,
            "has_glare": glare_ratio > 0.08,
            "too_dark": brightness < 60,
            "too_bright": brightness > 210
        }
    except Exception:
        return {"brightness": 128.0, "glare_ratio": 0.0, "has_glare": False,
                "too_dark": False, "too_bright": False}


# =============================================================================
# LIVE FACE DETECTOR (FOR CAPTURE GUIDANCE)
# =============================================================================

class LiveFaceDetector:
    """
    Fast, real-time face detector used ONLY to confirm a face is actually
    visible on the NIC photo while frames are being scanned during capture.
    This is deliberately separate from FaceRecognitionAgent (below), which
    does the heavier face-matching/embedding work after an image is captured.

    Backend priority (best available is picked automatically at startup):
      1. MediaPipe Face Detection - most accurate/fastest option here, installs
         as a plain Python wheel (no C++ build tools needed, unlike dlib), and
         is the recommended fix if face detection has been unreliable so far.
      2. OpenCV YuNet (DNN, ONNX)  - ships with recent opencv-python, no extra
         install beyond the one-time small model download.
      3. OpenCV Haar Cascade       - bundled inside every cv2 install, works
         fully offline, used only as a last-resort fallback since it is the
         least accurate of the three, especially at odd angles or low light.
    """

    YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
    YUNET_MODEL_URL = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )

    def __init__(self):
        self.backend = None
        self.detector = None
        self._mp_face = None
        self._haar = None
        self._input_size = (320, 320)
        self._init_backend()

    def _init_backend(self):
        # 1. MediaPipe
        if MEDIAPIPE_AVAILABLE:
            try:
                self._mp_face = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # 1 = full-range model, better for arm's-length ID capture
                    min_detection_confidence=MIN_FACE_CONFIDENCE
                )
                self.backend = "mediapipe"
                log_message("Live face detector: using MediaPipe", "INFO")
                return
            except Exception:
                pass

        # 2. OpenCV YuNet (DNN)
        try:
            model_path = self._ensure_yunet_model()
            if model_path:
                self.detector = cv2.FaceDetectorYN_create(
                    model_path, "", self._input_size,
                    score_threshold=MIN_FACE_CONFIDENCE, nms_threshold=0.3, top_k=5000
                )
                self.backend = "yunet"
                log_message("Live face detector: using OpenCV YuNet", "INFO")
                return
        except Exception:
            pass

        # 3. Haar Cascade (always available, no download required)
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if not self._haar.empty():
                self.backend = "haar"
                log_message("Live face detector: using Haar Cascade (fallback)", "WARN")
                return
        except Exception:
            pass

        self.backend = None
        log_message("Live face detector: no backend available - install mediapipe", "ERROR")

    def _ensure_yunet_model(self):
        """Download the small YuNet ONNX model once and cache it next to this script."""
        try:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, self.YUNET_MODEL_FILENAME)
            if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                return model_path
            resp = requests.get(self.YUNET_MODEL_URL, timeout=10)
            if resp.status_code == 200:
                with open(model_path, "wb") as f:
                    f.write(resp.content)
                return model_path
        except Exception:
            pass
        return None

    def detect(self, frame):
        """
        Detect faces in a BGR frame.
        Returns a list of {"bbox": (x, y, w, h), "confidence": float}, best-first.
        """
        if self.backend is None or frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]

        try:
            if self.backend == "mediapipe":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_face.process(rgb)
                faces = []
                if results.detections:
                    for det in results.detections:
                        box = det.location_data.relative_bounding_box
                        x = max(0, int(box.xmin * w))
                        y = max(0, int(box.ymin * h))
                        bw = max(1, int(box.width * w))
                        bh = max(1, int(box.height * h))
                        conf = float(det.score[0]) if det.score else 0.0
                        faces.append({"bbox": (x, y, bw, bh), "confidence": conf})
                faces.sort(key=lambda f: f["confidence"], reverse=True)
                return faces

            elif self.backend == "yunet":
                self.detector.setInputSize((w, h))
                _, results = self.detector.detect(frame)
                faces = []
                if results is not None:
                    for face in results:
                        x, y, bw, bh = face[0:4].astype(int)
                        conf = float(face[-1])
                        faces.append({"bbox": (max(0, int(x)), max(0, int(y)), int(bw), int(bh)),
                                      "confidence": conf})
                faces.sort(key=lambda f: f["confidence"], reverse=True)
                return faces

            elif self.backend == "haar":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                detections = self._haar.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
                )
                return [{"bbox": (int(x), int(y), int(bw), int(bh)), "confidence": 0.5}
                        for (x, y, bw, bh) in detections]

        except Exception:
            return []

        return []

    def is_available(self):
        return self.backend is not None


# Global, lazily-created live face detector shared across capture sessions
_live_face_detector = None

def get_live_face_detector():
    global _live_face_detector
    if _live_face_detector is None:
        _live_face_detector = LiveFaceDetector()
    return _live_face_detector


# =============================================================================
# AI FACE RECOGNITION FUNCTIONS
# =============================================================================

class FaceRecognitionAgent:
    """AI Face Recognition Agent for NIC Verification"""
    
    def __init__(self):
        self.model_name = "face_recognition"
        self.initialized = False
        self.face_app = None
        self._initialize()
    
    def _initialize(self):
        """Initialize face recognition models"""
        if INSIGHTFACE_AVAILABLE:
            try:
                self.face_app = FaceAnalysis(name='buffalo_l')
                self.face_app.prepare(ctx_id=0, det_size=(640, 640))
                self.model_name = "insightface"
                self.initialized = True
                return
            except Exception:
                pass
        
        if FACE_RECOGNITION_AVAILABLE:
            self.model_name = "face_recognition_dlib"
            self.initialized = True
            return
    
    def is_available(self):
        return self.initialized
    
    def detect_faces(self, image):
        """Detect faces in image"""
        if not self.initialized:
            return []
        
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image[0, 0, 0] > image[0, 0, 2]:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image
            else:
                rgb_image = image
            
            if self.model_name == "insightface" and self.face_app:
                faces = self.face_app.get(rgb_image)
                return faces
            
            elif self.model_name == "face_recognition_dlib" and FACE_RECOGNITION_AVAILABLE:
                face_locations = face_recognition.face_locations(rgb_image)
                face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                
                result = []
                for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
                    top, right, bottom, left = location
                    result.append({
                        'bbox': (left, top, right, bottom),
                        'encoding': encoding,
                        'index': i
                    })
                return result
            
            return []
        except Exception:
            return []
    
    def get_face_embedding(self, image, face_location=None):
        """Extract face embedding"""
        if not self.initialized:
            return None
        
        try:
            if self.model_name == "insightface" and self.face_app:
                faces = self.face_app.get(image)
                if faces:
                    return faces[0].embedding
                return None
            
            elif self.model_name == "face_recognition_dlib" and FACE_RECOGNITION_AVAILABLE:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    if image[0, 0, 0] > image[0, 0, 2]:
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_image = image
                else:
                    rgb_image = image
                
                if face_location:
                    top, right, bottom, left = face_location
                    encodings = face_recognition.face_encodings(rgb_image, [(top, right, bottom, left)])
                else:
                    encodings = face_recognition.face_encodings(rgb_image)
                
                if encodings:
                    return encodings[0]
                return None
            
            return None
        except Exception:
            return None
    
    def compare_faces(self, embedding1, embedding2):
        """Compare two face embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            similarity = max(0, (cosine_similarity + 1) / 2 * 100)
            
            # np.dot()/np.linalg.norm() return NumPy scalar types (np.float64).
            # Casting to a plain Python float here stops that scalar type from
            # propagating into 'similarity >= threshold' below (which would
            # otherwise yield np.bool_ - and on NumPy 2.x, np.bool_'s class
            # name is literally "bool", causing jsonify()/json.dumps() to
            # fail with the confusing "Object of type bool is not JSON
            # serializable" error).
            return float(similarity)
        except Exception:
            return 0.0
    
    def verify_face_match(self, image1, image2, threshold=0.6):
        """Verify if faces in two images match"""
        if not self.initialized:
            return {
                'success': False,
                'error': 'Face recognition not initialized',
                'similarity': 0,
                'match': False
            }
        
        try:
            emb1 = self.get_face_embedding(image1)
            emb2 = self.get_face_embedding(image2)
            
            if emb1 is None:
                return {
                    'success': False,
                    'error': 'No face detected in image 1',
                    'similarity': 0,
                    'match': False
                }
            
            if emb2 is None:
                return {
                    'success': False,
                    'error': 'No face detected in image 2',
                    'similarity': 0,
                    'match': False
                }
            
            similarity = self.compare_faces(emb1, emb2)
            
            return {
                'success': True,
                'similarity': similarity,
                'match': similarity >= (threshold * 100),
                'model': self.model_name
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'similarity': 0,
                'match': False
            }

# Create global face recognition agent
face_agent = None

def get_face_agent():
    """Get or create face recognition agent"""
    global face_agent
    if face_agent is None:
        face_agent = FaceRecognitionAgent()
    return face_agent

# =============================================================================
# API CALLS
# =============================================================================

def search_lime_documents(search_value):
    """Search for documents in LIME"""
    url = LIME_SEARCH_URL.format(search_value)
    headers = {"accept": "application/json", "Authorization": LIME_AUTH_TOKEN}

    response = requests.get(url, headers=headers, timeout=60)

    if response.status_code == 404:
        raise NoDocumentsFoundException("No documents found")

    if not response.ok:
        # Surface LIME's actual explanation (body/headers) instead of the
        # bare "403 Client Error: Forbidden" requests.raise_for_status()
        # gives - that string alone doesn't say WHY (bad/expired token,
        # missing 'Bearer ' scheme, IP not whitelisted for this gateway,
        # account not authorized, wrong host/env, etc).
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:500]
        raise PRDAPIError(
            f"LIME rejected the request ({response.status_code}): {detail} "
            f"[url={url}, auth_header_set={bool(LIME_AUTH_TOKEN)}, "
            f"www_authenticate={response.headers.get('WWW-Authenticate')}]"
        )

    return response.json()

def search_lime_documents_by_doctype(account_no, doc_type):
    """Search LIME for documents of a specific DocType tied to an account
    identifier (NIC or mobile). This targets the document/search/docType
    endpoint, distinct from the generic document/search used by
    search_lime_documents() above - the generic search has been observed
    to miss KYC/NIC documents that ARE visible via LIME's own portal
    "Document View By Any Field" search.

    The exact parameter names for this endpoint are unconfirmed (no
    Swagger access) - this follows the AccountNo query-param convention
    used by document/search, with DocType added. If that's wrong, the
    full response is captured in PRDAPIError so the mismatch is visible
    immediately rather than silently returning nothing.
    """
    url = f"{LIME_SEARCH_BY_DOCTYPE_URL}?AccountNo={account_no}&DocType={doc_type}"
    headers = {"accept": "application/json", "Authorization": LIME_AUTH_TOKEN}

    response = requests.get(url, headers=headers, timeout=60)

    if response.status_code == 404:
        raise NoDocumentsFoundException("No documents found")

    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:500]
        raise PRDAPIError(
            f"LIME docType search rejected the request ({response.status_code}): {detail} "
            f"[url={url}, auth_header_set={bool(LIME_AUTH_TOKEN)}]"
        )

    return response.json()

def download_lime_document(download_key):
    """Download a document from LIME"""
    url = LIME_DOWNLOAD_URL.format(download_key)
    headers = {"accept": "application/json", "Authorization": LIME_AUTH_TOKEN}

    response = requests.get(url, headers=headers, timeout=60)

    if not response.ok:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:500]
        raise PRDAPIError(
            f"LIME rejected the download request ({response.status_code}): {detail} "
            f"[url={url}, auth_header_set={bool(LIME_AUTH_TOKEN)}]"
        )

    return response.json()


# Metadata keys (case-insensitive) that may hold each field in the LIME
# document's MetaData array. Different LIME environments/doc templates have
# been observed using slightly different key names, so each field checks a
# small list of aliases. Adjust these lists if your LIME doc type uses a
# different key.
LIME_META_NIC_KEYS = ["nicnumber", "nic_number", "nic no", "nicno"]
LIME_META_NAME_KEYS = ["customername", "customer_name", "name", "fullname"]
LIME_META_ADDRESS_KEYS = [
    "address", "customeraddress", "customer_address",
    "permanentaddress", "permanent_address", "nicaddress"
]
LIME_META_MOBILE_KEYS = [
    "mobile", "mobileno", "mobile_no", "mobilenumber", "mobile_number",
    "contactno", "contact_no", "contactnumber", "phone", "phoneno", "phone_no"
]

def looks_like_nic(value):
    """Loose sanity check used only to decide whether a NIC-ish value found
    in LIME metadata is worth trusting, before falling back to other
    sources. Deliberately looser than is_valid_nic_format (which is the
    real gate right before calling PRD) - this just filters out obvious
    garbage like a single letter, so we don't lock in a bad value via the
    'and not nic_number' guard below and skip the fallback extractor."""
    if not value:
        return False
    digit_count = sum(ch.isdigit() for ch in str(value))
    return digit_count >= 9

def find_lime_documents_and_metadata(nic_number, mobile_number=None):
    """Search LIME for a customer's front NIC/KYC document and metadata,
    using the NIC as the sole search key (mobile number is NOT used to
    search LIME - it's optional and, if given, only cross-checked against
    whatever mobile number turns up in the matched document's metadata).

    Tries, in order:
      1. The generic document/search endpoint, by NIC.
      2. The DocType-scoped search endpoint, by NIC, for NIC_FRONT and
         KYC_FRONT specifically - catches documents (e.g. ones attached to
         a SIM-change or other transaction) that the generic search above
         has been observed to miss.

    Every attempt's outcome is recorded in "attempts" so a caller can build
    a fully transparent error message without digging through log files.

    Returns:
        {
            "front_doc": dict or None - the best matching document, if any
            "lime_meta": {"nic":..., "name":..., "address":..., "mobile":...}
            "lime_nic": normalized NIC string or None
            "matched_via": "nic" | "nic_docType_<TYPE>" | None
            "mobile_provided": the mobile number passed in (raw), or None
            "lime_mobile": mobile number found in LIME metadata, or None
            "mobile_match": True/False if both mobile_provided and
                lime_mobile are present and comparable, else None
            "attempts": [ {"key":..., "value": ..., "status":
                "found"/"empty"/"no_docs"/"error", "total_docs": N,
                "doc_types": [...], "error": str or None}, ... ]
        }
    """
    attempts = []
    front_doc = None
    lime_meta = {}
    lime_nic = None
    matched_via = None

    clean_nic = clean_nic_for_api(nic_number) if nic_number else None
    valid_nic = clean_nic if (clean_nic and is_valid_nic_format(clean_nic)) else None

    if not valid_nic:
        return {
            "front_doc": None, "lime_meta": {}, "lime_nic": None, "matched_via": None,
            "mobile_provided": mobile_number, "lime_mobile": None, "mobile_match": None,
            "attempts": [],
        }

    # ---- 1. Generic document/search by NIC --------------------------------
    try:
        response = search_lime_documents(valid_nic)
        docs = response.get("Documents", [])
        nic_docs = extract_nic_documents_from_lime(response)
        candidate = nic_docs.get("front") or (
            nic_docs["all_nic_docs"][0] if nic_docs.get("all_nic_docs") else None
        )
        doc_types = [f"{d.get('DocType', '?')} ({d.get('FileName', '?')})" for d in docs]
        attempts.append({
            "key": "nic", "value": valid_nic,
            "status": "found" if candidate else ("empty" if docs else "no_docs"),
            "total_docs": len(docs), "doc_types": doc_types, "error": None
        })

        meta = extract_metadata(response)
        for field in ("nic", "name", "address", "mobile"):
            if meta.get(field) and not lime_meta.get(field):
                lime_meta[field] = meta[field]
        if not lime_nic:
            candidate_nic = meta.get("nic") or extract_nic_from_documents(response)
            if candidate_nic:
                lime_nic = normalize_nic(candidate_nic)

        if candidate:
            front_doc = candidate
            matched_via = "nic"
    except NoDocumentsFoundException:
        attempts.append({"key": "nic", "value": valid_nic, "status": "no_docs",
                          "total_docs": 0, "doc_types": [], "error": None})
    except Exception as e:
        attempts.append({"key": "nic", "value": valid_nic, "status": "error",
                          "total_docs": 0, "doc_types": [], "error": str(e)})
        log_message(f"LIME search by nic ({valid_nic}) failed: {e}", "WARNING")

    # ---- 2. DocType-scoped search by NIC, if step 1 found nothing ---------
    if not front_doc:
        for doc_type in ("NIC_FRONT", "KYC_FRONT"):
            try:
                response = search_lime_documents_by_doctype(valid_nic, doc_type)
                docs = response.get("Documents", [])
                nic_docs = extract_nic_documents_from_lime(response)
                candidate = nic_docs.get("front") or (
                    nic_docs["all_nic_docs"][0] if nic_docs.get("all_nic_docs") else None
                )
                doc_types_seen = [f"{d.get('DocType', '?')} ({d.get('FileName', '?')})" for d in docs]
                attempts.append({
                    "key": f"nic (docType={doc_type})", "value": valid_nic,
                    "status": "found" if candidate else ("empty" if docs else "no_docs"),
                    "total_docs": len(docs), "doc_types": doc_types_seen, "error": None
                })

                meta = extract_metadata(response)
                for field in ("nic", "name", "address", "mobile"):
                    if meta.get(field) and not lime_meta.get(field):
                        lime_meta[field] = meta[field]
                if not lime_nic:
                    candidate_nic = meta.get("nic") or extract_nic_from_documents(response)
                    if candidate_nic:
                        lime_nic = normalize_nic(candidate_nic)

                if candidate:
                    front_doc = candidate
                    matched_via = f"nic_docType_{doc_type}"
                    break
            except NoDocumentsFoundException:
                attempts.append({"key": f"nic (docType={doc_type})", "value": valid_nic,
                                  "status": "no_docs", "total_docs": 0, "doc_types": [], "error": None})
            except Exception as e:
                attempts.append({"key": f"nic (docType={doc_type})", "value": valid_nic,
                                  "status": "error", "total_docs": 0, "doc_types": [], "error": str(e)})
                log_message(f"LIME docType search (nic={valid_nic}, docType={doc_type}) failed: {e}", "WARNING")

    # ---- Mobile is never used to search - only cross-checked -------------
    lime_mobile = lime_meta.get("mobile")
    mobile_match = None
    if mobile_number and lime_mobile:
        try:
            mobile_match = format_mobile_number(mobile_number) == format_mobile_number(lime_mobile)
        except Exception:
            mobile_match = str(mobile_number).strip() == str(lime_mobile).strip()

    return {
        "front_doc": front_doc,
        "lime_meta": lime_meta,
        "lime_nic": lime_nic,
        "matched_via": matched_via,
        "mobile_provided": mobile_number,
        "lime_mobile": lime_mobile,
        "mobile_match": mobile_match,
        "attempts": attempts,
    }

def find_lime_documents_via_mobile(mobile_number, nic_number=None):
    """Search LIME for a customer's front NIC/KYC document, using the
    MOBILE NUMBER as the primary search key (falling back to NIC, then a
    DocType-scoped NIC search) - used by the two modes where an image is
    already supplied directly (stored/captured image + LIME + PRD) and the
    mobile number is the natural, required lookup key. Distinct from
    find_lime_documents_and_metadata, which is NIC-primary/mobile-optional
    and used by the LIME-repository-only mode.

    Returns the same shape as find_lime_documents_and_metadata.
    """
    attempts = []
    front_doc = None
    lime_meta = {}
    lime_nic = None
    matched_via = None

    formatted_mobile = format_mobile_number(mobile_number) if mobile_number else None
    clean_nic = clean_nic_for_api(nic_number) if nic_number else None
    valid_nic = clean_nic if (clean_nic and is_valid_nic_format(clean_nic)) else None

    search_order = []
    if formatted_mobile:
        search_order.append(("mobile", formatted_mobile))
    if valid_nic:
        search_order.append(("nic", valid_nic))

    for key_type, key_value in search_order:
        try:
            response = search_lime_documents(key_value)
            docs = response.get("Documents", [])
            nic_docs = extract_nic_documents_from_lime(response)
            candidate = nic_docs.get("front") or (
                nic_docs["all_nic_docs"][0] if nic_docs.get("all_nic_docs") else None
            )
            doc_types = [f"{d.get('DocType', '?')} ({d.get('FileName', '?')})" for d in docs]
            attempts.append({
                "key": key_type, "value": key_value,
                "status": "found" if candidate else ("empty" if docs else "no_docs"),
                "total_docs": len(docs), "doc_types": doc_types, "error": None
            })

            meta = extract_metadata(response)
            for field in ("nic", "name", "address", "mobile"):
                if meta.get(field) and not lime_meta.get(field):
                    lime_meta[field] = meta[field]
            if not lime_nic:
                candidate_nic = meta.get("nic") or extract_nic_from_documents(response)
                if candidate_nic:
                    lime_nic = normalize_nic(candidate_nic)

            if candidate and not front_doc:
                front_doc = candidate
                matched_via = key_type
        except NoDocumentsFoundException:
            attempts.append({"key": key_type, "value": key_value, "status": "no_docs",
                              "total_docs": 0, "doc_types": [], "error": None})
        except Exception as e:
            attempts.append({"key": key_type, "value": key_value, "status": "error",
                              "total_docs": 0, "doc_types": [], "error": str(e)})
            log_message(f"LIME search by {key_type} ({key_value}) failed: {e}", "WARNING")

    if not front_doc:
        doctype_candidates = ["NIC_FRONT", "KYC_FRONT"]
        identity_candidates = [(t, v) for t, v in [("mobile", formatted_mobile), ("nic", valid_nic)] if v]
        for key_type, key_value in identity_candidates:
            if front_doc:
                break
            for doc_type in doctype_candidates:
                try:
                    response = search_lime_documents_by_doctype(key_value, doc_type)
                    docs = response.get("Documents", [])
                    nic_docs = extract_nic_documents_from_lime(response)
                    candidate = nic_docs.get("front") or (
                        nic_docs["all_nic_docs"][0] if nic_docs.get("all_nic_docs") else None
                    )
                    doc_types_seen = [f"{d.get('DocType', '?')} ({d.get('FileName', '?')})" for d in docs]
                    attempts.append({
                        "key": f"{key_type} (docType={doc_type})", "value": key_value,
                        "status": "found" if candidate else ("empty" if docs else "no_docs"),
                        "total_docs": len(docs), "doc_types": doc_types_seen, "error": None
                    })
                    meta = extract_metadata(response)
                    for field in ("nic", "name", "address", "mobile"):
                        if meta.get(field) and not lime_meta.get(field):
                            lime_meta[field] = meta[field]
                    if not lime_nic:
                        candidate_nic = meta.get("nic") or extract_nic_from_documents(response)
                        if candidate_nic:
                            lime_nic = normalize_nic(candidate_nic)
                    if candidate:
                        front_doc = candidate
                        matched_via = f"{key_type}_docType_{doc_type}"
                        break
                except NoDocumentsFoundException:
                    attempts.append({"key": f"{key_type} (docType={doc_type})", "value": key_value,
                                      "status": "no_docs", "total_docs": 0, "doc_types": [], "error": None})
                except Exception as e:
                    attempts.append({"key": f"{key_type} (docType={doc_type})", "value": key_value,
                                      "status": "error", "total_docs": 0, "doc_types": [], "error": str(e)})
                    log_message(f"LIME docType search ({key_type}={key_value}, docType={doc_type}) failed: {e}", "WARNING")

    return {
        "front_doc": front_doc,
        "lime_meta": lime_meta,
        "lime_nic": lime_nic,
        "matched_via": matched_via,
        "formatted_mobile": formatted_mobile,
        "attempts": attempts,
    }

def format_lime_search_diagnostics(lime_result):
    """Turn find_lime_documents_and_metadata()'s attempts into a readable
    on-screen explanation - no log file needed to see what happened."""
    if not lime_result["attempts"]:
        return "No valid NIC was provided to search LIME with."
    parts = []
    for a in lime_result["attempts"]:
        label = f"{a['key']}={a['value']}"
        if a["status"] == "no_docs":
            parts.append(f"{label}: LIME returned 0 documents.")
        elif a["status"] == "error":
            parts.append(f"{label}: search failed ({a['error']}).")
        elif a["status"] == "empty":
            parts.append(
                f"{label}: {a['total_docs']} document(s) returned, none matched a "
                f"recognized NIC/KYC DocType (recognized: {LIME_DOC_TYPES}). "
                f"Actual DocTypes: {a['doc_types']}"
            )
        elif a["status"] == "found":
            parts.append(f"{label}: found a matching document.")
    return " ".join(parts)

def build_lime_metadata_debug(search_response):
    """Every raw DocType/FileName/MetaData key-value pair LIME returned,
    for surfacing directly in API responses when a field like name/nic
    doesn't extract as expected - avoids needing to dig through
    verification_logs.txt to see what LIME actually sent."""
    if not search_response:
        return []
    debug = []
    for doc in search_response.get("Documents", []):
        debug.append({
            "DocType": doc.get("DocType"),
            "FileName": doc.get("FileName"),
            "MetaData": [
                {"Key": item.get("Key"), "Value": item.get("Value")}
                for item in doc.get("MetaData", [])
            ]
        })
    return debug

def extract_metadata(search_response):
    """Extract NIC, name, address and mobile from LIME document metadata"""
    nic_number = None
    customer_name = None
    customer_address = None
    customer_mobile = None
    all_metadata_seen = []  # only used for the debug dump below if nothing matches

    for doc in search_response.get("Documents", []):
        for item in doc.get("MetaData", []):
            raw_key = item.get("Key", "")
            key = str(raw_key).strip().lower()
            value = item.get("Value")
            all_metadata_seen.append({"key": raw_key, "value": value, "doc": doc.get("FileName")})
            if not value:
                continue
            if key in LIME_META_NIC_KEYS and not nic_number:
                if looks_like_nic(value):
                    nic_number = value
                else:
                    # Don't lock in obvious garbage (e.g. a single letter) -
                    # leave nic_number unset so the caller's fallback to
                    # extract_nic_from_documents gets a chance, and log the
                    # raw metadata so this can be traced back to the actual
                    # LIME key/value causing it.
                    log_message(
                        f"Ignoring implausible NIC value from LIME metadata: "
                        f"key='{item.get('Key')}' value='{value}' "
                        f"doc_filename='{doc.get('FileName')}' "
                        f"raw_metadata={doc.get('MetaData')}",
                        "WARNING"
                    )
            elif key in LIME_META_NAME_KEYS and not customer_name:
                customer_name = value
            elif key in LIME_META_ADDRESS_KEYS and not customer_address:
                customer_address = value
            elif key in LIME_META_MOBILE_KEYS and not customer_mobile:
                customer_mobile = value

    if not nic_number:
        # None of the known key names (LIME_META_NIC_KEYS) matched at all -
        # try a looser pass: any metadata key that merely *contains* "nic",
        # gated by looks_like_nic so it can't grab an unrelated field. This
        # environment/doc template may simply use a key name we don't have
        # in the alias list yet.
        for doc in search_response.get("Documents", []):
            for item in doc.get("MetaData", []):
                key = str(item.get("Key", "")).strip().lower()
                value = item.get("Value")
                if value and "nic" in key and looks_like_nic(value):
                    nic_number = value
                    log_message(
                        f"Found NIC via loose key match (not in LIME_META_NIC_KEYS): "
                        f"key='{item.get('Key')}' value='{value}' - consider adding "
                        f"'{key}' to LIME_META_NIC_KEYS.",
                        "INFO"
                    )
                    break
            if nic_number:
                break

    if not nic_number:
        # Still nothing - dump every raw Key/Value this LIME response
        # actually contained, so the real key name (whatever it is) can be
        # read straight from the log instead of guessing.
        log_message(
            f"No NIC found in LIME metadata under known keys {LIME_META_NIC_KEYS} "
            f"or any loose 'nic'-containing key. Raw metadata seen: {all_metadata_seen}",
            "WARNING"
        )

    if not customer_name:
        # Same gap as the NIC field can have: the exact known key names
        # (LIME_META_NAME_KEYS) may not match this environment's actual key.
        # Try a looser pass on any key containing "name" (but not "file",
        # "doc", or "customer_name" variants we might have half-matched -
        # those are already covered above) before giving up.
        for doc in search_response.get("Documents", []):
            for item in doc.get("MetaData", []):
                key = str(item.get("Key", "")).strip().lower()
                value = item.get("Value")
                if value and "name" in key and "file" not in key:
                    customer_name = value
                    log_message(
                        f"Found name via loose key match (not in LIME_META_NAME_KEYS): "
                        f"key='{item.get('Key')}' value='{value}' - consider adding "
                        f"'{key}' to LIME_META_NAME_KEYS.",
                        "INFO"
                    )
                    break
            if customer_name:
                break
        if not customer_name:
            log_message(
                f"No name found in LIME metadata under known keys {LIME_META_NAME_KEYS} "
                f"or any loose 'name'-containing key.",
                "WARNING"
            )

    if not customer_mobile:
        # Same loose-match treatment as name/nic - try any key containing
        # "mobile"/"contact"/"phone" before giving up.
        for doc in search_response.get("Documents", []):
            for item in doc.get("MetaData", []):
                key = str(item.get("Key", "")).strip().lower()
                value = item.get("Value")
                if value and any(alias in key for alias in ("mobile", "contact", "phone")):
                    customer_mobile = value
                    log_message(
                        f"Found mobile via loose key match (not in LIME_META_MOBILE_KEYS): "
                        f"key='{item.get('Key')}' value='{value}' - consider adding "
                        f"'{key}' to LIME_META_MOBILE_KEYS.",
                        "INFO"
                    )
                    break
            if customer_mobile:
                break

    return {"nic": nic_number, "name": customer_name, "address": customer_address, "mobile": customer_mobile}

def extract_nic_from_documents(search_response):
    """Try to find NIC from document filenames or other fields"""
    nic_pattern_re = re.compile(r'\b\d{9}[Vv]\b|\b\d{12}\b')
    for doc in search_response.get("Documents", []):
        filename = doc.get("FileName", "")
        match = nic_pattern_re.search(filename)
        if match:
            return match.group()
        
        for item in doc.get("MetaData", []):
            value = str(item.get("Value", "")).strip()
            if nic_pattern_re.fullmatch(value):
                return value
    
    return None

def extract_nic_documents_from_lime(search_response):
    """Extract NIC front documents from LIME response"""
    front_docs = []
    back_docs = []
    all_nic_docs = []
    seen_doc_types = []
    all_docs_seen = []  # every DocType/FileName present, regardless of match - for the empty-match log below

    for doc in search_response.get("Documents", []):
        raw_doc_type = doc.get("DocType", "")
        doc_type = raw_doc_type.upper().replace("_", " ").replace("-", " ").strip()
        all_docs_seen.append({"DocType": raw_doc_type, "FileName": doc.get("FileName")})

        is_nic_doc = False
        for nic_type in LIME_DOC_TYPES:
            if nic_type.upper() in doc_type or nic_type in doc_type:
                is_nic_doc = True
                break

        if is_nic_doc:
            seen_doc_types.append(raw_doc_type)
            all_nic_docs.append(doc)
            
            if "FRONT" in doc_type:
                front_docs.append(doc)
            elif "BACK" in doc_type:
                back_docs.append(doc)
            
            if "KYC" in doc_type and "FRONT" in doc_type:
                front_docs.append(doc)
            elif "NIC" in doc_type and "FRONT" in doc_type:
                front_docs.append(doc)
    
    front_docs.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
    back_docs.sort(key=lambda x: x.get("createdAt", ""), reverse=True)
    all_nic_docs.sort(key=lambda x: x.get("createdAt", ""), reverse=True)

    if not front_docs and all_nic_docs:
        # A NIC/KYC document exists but none of them literally has "FRONT"
        # in the DocType (e.g. it's just typed "NIC" or "KYC" with no
        # front/back distinction - common for single-sided scans). Log the
        # actual DocType values so this can be traced, since LIME_DOC_TYPES
        # or this matching logic may need to be extended for this doc type.
        log_message(
            f"No document with 'FRONT' in DocType found, but {len(all_nic_docs)} "
            f"NIC/KYC-ish doc(s) exist. Raw DocTypes seen: {seen_doc_types}",
            "WARNING"
        )
    elif not all_nic_docs:
        # Nothing at all matched LIME_DOC_TYPES - either this account has no
        # documents, or its DocType values don't contain any of our known
        # NIC/KYC aliases. Dump every document actually present (DocType +
        # FileName) so we can see whether it's a real gap in LIME_DOC_TYPES
        # or genuinely no NIC document for this account.
        log_message(
            f"No document matched any LIME_DOC_TYPES alias {LIME_DOC_TYPES}. "
            f"All documents present in this LIME response: {all_docs_seen}",
            "WARNING"
        )

    return {
        "front": front_docs[0] if front_docs else None,
        "back": back_docs[0] if back_docs else None,
        "all_nic_docs": all_nic_docs
    }


def extract_signature_documents_from_lime(search_response):
    """Extract the customer's stored SIGNATURE document(s) from a LIME
    document/search response - the signature-specific counterpart to
    extract_nic_documents_from_lime() above. Matches DocType against
    LIME_SIGNATURE_DOC_TYPES the same way NIC/KYC docs are matched against
    LIME_DOC_TYPES, then returns the most recently created one."""
    signature_docs = []
    seen_doc_types = []
    all_docs_seen = []

    for doc in search_response.get("Documents", []):
        raw_doc_type = doc.get("DocType", "")
        doc_type = raw_doc_type.upper().replace("_", " ").replace("-", " ").strip()
        all_docs_seen.append({"DocType": raw_doc_type, "FileName": doc.get("FileName")})

        is_signature_doc = False
        for sig_type in LIME_SIGNATURE_DOC_TYPES:
            normalized_sig_type = sig_type.upper().replace("_", " ").replace("-", " ").strip()
            if normalized_sig_type in doc_type:
                is_signature_doc = True
                break

        if is_signature_doc:
            seen_doc_types.append(raw_doc_type)
            signature_docs.append(doc)

    signature_docs.sort(key=lambda x: x.get("createdAt", ""), reverse=True)

    if not signature_docs:
        log_message(
            f"No document matched any LIME_SIGNATURE_DOC_TYPES alias {LIME_SIGNATURE_DOC_TYPES}. "
            f"All documents present in this LIME response: {all_docs_seen}",
            "WARNING"
        )

    return {
        "signature": signature_docs[0] if signature_docs else None,
        "all_signature_docs": signature_docs
    }


def find_lime_signature_document(nic_number, mobile_number=None):
    """Search LIME for a customer's stored SIGNATURE document, using the
    NIC as the sole search key - the signature-specific counterpart to
    find_lime_documents_and_metadata(). Mobile is never used to search
    LIME, only cross-checked against the mobile found in the matched
    document's own metadata, exactly as the NIC/KYC lookup does.

    Tries, in order:
      1. The generic document/search endpoint, by NIC.
      2. The DocType-scoped search endpoint, by NIC, for each alias in
         LIME_SIGNATURE_DOC_TYPES - catches signature docs the generic
         search misses (same reasoning as the NIC_FRONT/KYC_FRONT
         DocType-scoped fallback above).

    Returns the same shape as find_lime_documents_and_metadata(), with
    "signature_doc" in place of "front_doc".
    """
    attempts = []
    signature_doc = None
    lime_meta = {}
    lime_nic = None
    matched_via = None

    clean_nic = clean_nic_for_api(nic_number) if nic_number else None
    valid_nic = clean_nic if (clean_nic and is_valid_nic_format(clean_nic)) else None

    if not valid_nic:
        return {
            "signature_doc": None, "lime_meta": {}, "lime_nic": None, "matched_via": None,
            "mobile_provided": mobile_number, "lime_mobile": None, "mobile_match": None,
            "attempts": [],
        }

    # ---- 1. Generic document/search by NIC --------------------------------
    try:
        response = search_lime_documents(valid_nic)
        docs = response.get("Documents", [])
        sig_docs = extract_signature_documents_from_lime(response)
        candidate = sig_docs.get("signature")
        doc_types = [f"{d.get('DocType', '?')} ({d.get('FileName', '?')})" for d in docs]
        attempts.append({
            "key": "nic", "value": valid_nic,
            "status": "found" if candidate else ("empty" if docs else "no_docs"),
            "total_docs": len(docs), "doc_types": doc_types, "error": None
        })

        meta = extract_metadata(response)
        for field in ("nic", "name", "address", "mobile"):
            if meta.get(field) and not lime_meta.get(field):
                lime_meta[field] = meta[field]
        if not lime_nic:
            candidate_nic = meta.get("nic") or extract_nic_from_documents(response)
            if candidate_nic:
                lime_nic = normalize_nic(candidate_nic)

        if candidate:
            signature_doc = candidate
            matched_via = "nic"
    except NoDocumentsFoundException:
        attempts.append({"key": "nic", "value": valid_nic, "status": "no_docs",
                          "total_docs": 0, "doc_types": [], "error": None})
    except Exception as e:
        attempts.append({"key": "nic", "value": valid_nic, "status": "error",
                          "total_docs": 0, "doc_types": [], "error": str(e)})
        log_message(f"LIME signature search by nic ({valid_nic}) failed: {e}", "WARNING")

    # ---- 2. DocType-scoped search by NIC, if step 1 found nothing ---------
    if not signature_doc:
        for doc_type in LIME_SIGNATURE_DOC_TYPES:
            try:
                response = search_lime_documents_by_doctype(valid_nic, doc_type)
                docs = response.get("Documents", [])
                sig_docs = extract_signature_documents_from_lime(response)
                candidate = sig_docs.get("signature")
                doc_types_seen = [f"{d.get('DocType', '?')} ({d.get('FileName', '?')})" for d in docs]
                attempts.append({
                    "key": f"nic (docType={doc_type})", "value": valid_nic,
                    "status": "found" if candidate else ("empty" if docs else "no_docs"),
                    "total_docs": len(docs), "doc_types": doc_types_seen, "error": None
                })

                meta = extract_metadata(response)
                for field in ("nic", "name", "address", "mobile"):
                    if meta.get(field) and not lime_meta.get(field):
                        lime_meta[field] = meta[field]
                if not lime_nic:
                    candidate_nic = meta.get("nic") or extract_nic_from_documents(response)
                    if candidate_nic:
                        lime_nic = normalize_nic(candidate_nic)

                if candidate:
                    signature_doc = candidate
                    matched_via = f"nic_docType_{doc_type}"
                    break
            except NoDocumentsFoundException:
                attempts.append({"key": f"nic (docType={doc_type})", "value": valid_nic,
                                  "status": "no_docs", "total_docs": 0, "doc_types": [], "error": None})
            except Exception as e:
                attempts.append({"key": f"nic (docType={doc_type})", "value": valid_nic,
                                  "status": "error", "total_docs": 0, "doc_types": [], "error": str(e)})
                log_message(f"LIME signature docType search (nic={valid_nic}, docType={doc_type}) failed: {e}", "WARNING")

    lime_mobile = lime_meta.get("mobile")
    mobile_match = None
    if mobile_number and lime_mobile:
        try:
            mobile_match = format_mobile_number(mobile_number) == format_mobile_number(lime_mobile)
        except Exception:
            mobile_match = str(mobile_number).strip() == str(lime_mobile).strip()

    return {
        "signature_doc": signature_doc,
        "lime_meta": lime_meta,
        "lime_nic": lime_nic,
        "matched_via": matched_via,
        "mobile_provided": mobile_number,
        "lime_mobile": lime_mobile,
        "mobile_match": mobile_match,
        "attempts": attempts,
    }


def format_lime_signature_search_diagnostics(lime_result):
    """Signature-doc counterpart to format_lime_search_diagnostics() - builds
    a human-readable explanation of every LIME search attempt made while
    looking for a SIGNATURE document, for use in error messages."""
    lines = []
    for a in lime_result.get("attempts", []):
        if a["status"] == "found":
            continue
        if a["status"] == "no_docs":
            lines.append(f"  - {a['key']}={a['value']}: no documents found in LIME for this account.")
        elif a["status"] == "empty":
            lines.append(
                f"  - {a['key']}={a['value']}: {a['total_docs']} document(s) found, but none matched a "
                f"recognized SIGNATURE DocType (recognized: {LIME_SIGNATURE_DOC_TYPES}). "
                f"Actual DocTypes: {a['doc_types']}"
            )
        elif a["status"] == "error":
            lines.append(f"  - {a['key']}={a['value']}: search failed ({a['error']})")
    if not lines:
        return "No SIGNATURE document found in LIME for this NIC."
    return "No SIGNATURE document found in LIME for this NIC. Details:\n" + "\n".join(lines)


def extract_image_data(download_response):
    """Extract base64 image from download response"""
    for key in ["FileData", "document", "data", "base64"]:
        if key in download_response:
            return download_response[key]
    raise Exception("Cannot extract image data from download response")

def extract_prd_data(prd_response):
    """Extract customer data from PRD response"""
    data = prd_response.get("responseData", {})
    if not data:
        data = prd_response.get("nicResponseData", {})
    
    name_parts = [data.get("name1"), data.get("name2")]
    full_name = " ".join(filter(None, name_parts)) if any(name_parts) else None

    # Address in PRD responses has been seen either as a single field or
    # split across up to three lines - join whichever are present.
    if data.get("address"):
        full_address = data.get("address")
    else:
        address_parts = [
            data.get("addressLine1") or data.get("address1"),
            data.get("addressLine2") or data.get("address2"),
            data.get("addressLine3") or data.get("address3"),
        ]
        full_address = ", ".join(filter(None, address_parts)) if any(address_parts) else None
    
    return {
        "nic": data.get("latestIDCardNo") or data.get("previousIDCardNo") or data.get("idCardNumber"),
        "latestIDCardNo": data.get("latestIDCardNo"),
        "previousIDCardNo": data.get("previousIDCardNo"),
        "idCardNumber": data.get("idCardNumber"),
        "name": full_name,
        "address": full_address,
        "dob": data.get("dateOfBirth"),
        "issue_date": (
            data.get("idCardIssueDate")
            or data.get("issueDate")
            or data.get("dateOfIssue")
            or data.get("cardIssueDate")
            or data.get("nicIssueDate")
            or data.get("issuedDate")
        ),
        "gender": data.get("gender"),
        "phone": data.get("contactPhoneNo"),
        "nic_front_image": data.get("nicFrontImage"),
        "photograph_image": data.get("photographImage"),
        "nic_status": data.get("nicStatus")
    }

# =============================================================================
# LIME vs PRD DATA FIELD COMPARISON (name / NIC / address)
# =============================================================================

def _normalize_text(value):
    """Lowercase, strip and collapse whitespace for fuzzy text comparison"""
    if not value:
        return ""
    return re.sub(r'\s+', ' ', str(value).strip().lower())

def fuzzy_text_match(value_a, value_b):
    """Return a 0-100 similarity score between two free-text values"""
    a = _normalize_text(value_a)
    b = _normalize_text(value_b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio() * 100

def compare_customer_data(lime_meta, prd_data, lime_nic_normalized=None):
    """
    Compare customer identity fields (name, NIC, address) reported by LIME
    document metadata against the PRD record for the same customer.
    Returns per-field match info plus a combined data-match score (0-100).
    """
    fields = {}

    lime_nic = lime_nic_normalized or lime_meta.get("nic")
    prd_nic = prd_data.get("nic")
    nic_score = fuzzy_text_match(lime_nic, prd_nic) if (lime_nic and prd_nic) else 0.0
    fields["nic"] = {
        "lime_value": lime_nic,
        "prd_value": prd_nic,
        "score": nic_score,
        "match": nic_score >= 90  # NIC should match near-exactly once normalized
    }

    name_score = fuzzy_text_match(lime_meta.get("name"), prd_data.get("name"))
    fields["name"] = {
        "lime_value": lime_meta.get("name"),
        "prd_value": prd_data.get("name"),
        "score": name_score,
        "match": name_score >= 75
    }

    address_score = fuzzy_text_match(lime_meta.get("address"), prd_data.get("address"))
    fields["address"] = {
        "lime_value": lime_meta.get("address"),
        "prd_value": prd_data.get("address"),
        "score": address_score,
        "match": address_score >= 60  # addresses vary more in formatting/OCR
    }

    # Only average fields where both sides actually had a value to compare,
    # so a missing PRD/LIME field doesn't unfairly drag the score down.
    comparable_scores = [f["score"] for f in fields.values() if f["lime_value"] and f["prd_value"]]
    overall_score = sum(comparable_scores) / len(comparable_scores) if comparable_scores else 0.0

    return {
        "fields": fields,
        "overall_score": overall_score,
        "all_match": all(f["match"] for f in fields.values() if f["lime_value"] and f["prd_value"])
    }

def query_prd_customer(nic_no, requested_user="Chaminda_01930", nic_front_image=""):
    """Query PRD for customer data via validateNicByDrp.

    nic_front_image is optional (defaults to "") - existing callers that
    only pass nic_no are unaffected. No nicBackImage field: this endpoint
    doesn't need it.
    """
    url = PRD_QUERY_URL_TEMPLATE.format(nic=nic_no)
    headers = {
        "Content-Type": "application/json",
        "X-Channel-Key": PRD_CHANNEL_KEY
    }
    payload = {
        "channel": "LIME",
        "nicNo": nic_no,
        "requestedUser": requested_user,
        "nicFrontImage": nic_front_image
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if not response.ok:
            # Surface the PRD gateway's actual explanation instead of just
            # "400 Client Error" - it's almost always in the response body.
            try:
                body = response.json()
                detail = body.get("executionMessage") or body.get("message") or body
            except ValueError:
                detail = response.text[:500]
            raise PRDAPIError(
                f"PRD rejected the request ({response.status_code}): {detail} "
                f"[url={url}, channel_key_set={bool(PRD_CHANNEL_KEY)}]"
            )

        data = response.json()

        if data.get('executionCode') != '00':
            raise PRDAPIError(f"PRD API returned error: {data.get('executionMessage', 'Unknown error')}")
        
        return data
    except requests.exceptions.RequestException as e:
        raise PRDAPIError(f"Failed to query PRD: {str(e)}")
    except PRDAPIError:
        raise
    except Exception as e:
        raise PRDAPIError(f"Error processing PRD response: {str(e)}")

# =============================================================================
# AUTOMATED CAPTURE SYSTEM - ENHANCED WITH 10 SECOND TIMER
# =============================================================================

def open_webcam(max_index=4, warmup_reads=5):
    """
    Robustly open the laptop's webcam.

    On many Windows laptops, cv2.VideoCapture(0, cv2.CAP_DSHOW) reports the
    DSHOW backend as available but still fails with:
        "backend is generally available but can't be used to capture by index"
    This can happen because: the camera is already in use by another app
    (Teams/Zoom/Windows Camera/a browser tab), the camera is disabled in
    Windows privacy settings, the built-in webcam isn't index 0 on that
    machine, or the DSHOW backend itself doesn't get along with that
    particular driver.

    This function tries every combination of backend and index (0-3) and
    confirms a REAL frame can be read (isOpened() alone can be misleadingly
    True even when no frame can ever be grabbed) before returning.
    """
    if sys.platform.startswith("win"):
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "ANY"),
        ]
    else:
        backends = [(cv2.CAP_ANY, "ANY")]

    for index in range(max_index):
        for backend_flag, backend_name in backends:
            try:
                cam = cv2.VideoCapture(index, backend_flag)
            except Exception:
                continue

            if not cam.isOpened():
                cam.release()
                continue

            got_frame = False
            for _ in range(warmup_reads):
                try:
                    ret, frame = cam.read()
                except Exception:
                    ret, frame = False, None
                if ret and frame is not None and frame.size > 0:
                    got_frame = True
                    break
                time.sleep(0.1)

            if got_frame:
                log_message(f"Camera opened: index={index}, backend={backend_name}", "INFO")
                return cam, index, backend_name

            cam.release()

    return None, None, None


def print_camera_troubleshooting():
    print_warn("Could not open any webcam on this machine.")
    print("     Try the following, then run this again:")
    print("     • Close other apps that may be using the camera")
    print("       (Zoom, Microsoft Teams, Windows Camera app, browser video-call tabs)")
    print("     • Windows Settings > Privacy & Security > Camera")
    print("       - make sure 'Camera access' and 'Let apps access your camera' are ON")
    print("     • Device Manager > Cameras - confirm the webcam isn't disabled or showing a driver error")
    print("     • If it's an external USB webcam, unplug and reconnect it (try a different USB port)")
    print("     • Restart the laptop if the camera was recently used by another app that may not have released it")


class AutoNICCapture:
    """Automated NIC capture with quality verification and timeout - 10 seconds"""
    
    def __init__(self):
        self.camera = None
        self.is_capturing = False
        self.captured_image = None
        self.quality_score = 0
        self.capture_timer = None
        self.capture_complete = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.best_frame = None
        self.best_quality = 0
        self.best_size_check = None
        self.frames_analyzed = 0
        self.frames_with_nic = 0
        self.frames_with_face = 0
        self.quality_history = []
        self._best_has_face = False
        self._best_face_bbox = None
        self.face_detector = get_live_face_detector()
    
    def _calculate_frame_quality(self, frame):
        """Calculate comprehensive quality score for a frame, including live face detection and glare check"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Sharpness - more forgiving
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 200)  # Reduced denominator
            
            # NIC detection
            rect = detect_nic_region(frame)
            has_nic = rect is not None
            nic_score = 0.5 if has_nic else 0  # Reduced from 0.7
            
            # Edge density
            edge_density = detect_edges(frame)
            edge_score = min(1.0, edge_density * 3)  # More forgiving
            
            # Size check if NIC detected
            size_valid = False
            size_info = None
            if has_nic:
                size_info = check_nic_size_and_quality(frame)
                size_valid = size_info.get("valid", False)
                size_score = size_info.get("score", 0)
            else:
                size_score = 0
            
            # Face detection - search inside the NIC card region if we found
            # one (the photo lives there), otherwise fall back to the whole
            # frame so we still get feedback before the card is positioned
            search_region = rect if has_nic else None
            # A crop this small (e.g. a NIC region detected far from the camera)
            # gives the DNN-based detector nothing usable and just triggers
            # internal shape-mismatch warnings, so fall back to the full frame
            # instead of detecting on a sliver of pixels.
            MIN_FACE_SEARCH_DIM = 60
            if search_region and search_region[2] >= MIN_FACE_SEARCH_DIM and search_region[3] >= MIN_FACE_SEARCH_DIM:
                x, y, w, h = search_region
                sub_frame = frame[max(0, y):y + h, max(0, x):x + w]
                raw_faces = self.face_detector.detect(sub_frame) if sub_frame.size > 0 else []
                faces = [
                    {"bbox": (f["bbox"][0] + x, f["bbox"][1] + y, f["bbox"][2], f["bbox"][3]),
                     "confidence": f["confidence"]}
                    for f in raw_faces
                ]
            else:
                faces = self.face_detector.detect(frame)
            
            has_face = len(faces) > 0
            best_face = faces[0] if faces else None
            face_score = min(1.0, best_face["confidence"]) if best_face else 0
            
            # Glare / brightness check - laminated NIC cards commonly blow out
            # under indoor lighting, which is a frequent, silent cause of face
            # detection failing even when the frame otherwise looks sharp
            glare_info = detect_glare_and_brightness(frame, region=rect if has_nic else None)
            glare_penalty = 0.25 if glare_info["has_glare"] else 0
            brightness_penalty = 0.15 if (glare_info["too_dark"] or glare_info["too_bright"]) else 0
            
            # Combined quality - more forgiving weights, face detection now
            # carries the largest single weight since it is what verification
            # ultimately depends on
            quality = 0
            quality += sharpness_score * 0.20
            quality += nic_score * 0.20
            quality += edge_score * 0.10
            quality += (size_score * 0.20 if size_valid else 0)
            quality += face_score * 0.30
            quality -= glare_penalty
            quality -= brightness_penalty
            quality = max(0.0, min(1.0, quality))
            
            # Boost if NIC + face both detected even with otherwise low quality
            if has_nic and has_face and quality < 0.35:
                quality = max(quality, 0.35)
            
            # Track presence stats
            if has_nic:
                self.frames_with_nic += 1
            if has_face:
                self.frames_with_face += 1
            
            return {
                "quality": quality,
                "sharpness": sharpness,
                "has_nic": has_nic,
                "size_valid": size_valid,
                "size_info": size_info,
                "edge_density": edge_density,
                "has_face": has_face,
                "face_bbox": best_face["bbox"] if best_face else None,
                "face_confidence": best_face["confidence"] if best_face else 0,
                "glare_info": glare_info
            }
        except:
            return {"quality": 0.1, "has_nic": False, "size_valid": False,
                    "has_face": False, "face_bbox": None, "face_confidence": 0,
                    "glare_info": {"has_glare": False, "too_dark": False, "too_bright": False}}
    
    def _capture_worker(self):
        """Background worker for continuous frame capture"""
        while self.is_capturing:
            if self.camera is not None:
                ret, frame = self.camera.read()
                if ret:
                    quality_data = self._calculate_frame_quality(frame)
                    self.frames_analyzed += 1
                    
                    # Store quality history
                    self.quality_history.append(quality_data["quality"])
                    if len(self.quality_history) > 50:
                        self.quality_history.pop(0)
                    
                    # Update best frame if quality is better - a frame with a
                    # detected face always outranks one without, since a frame
                    # missing a face is unusable for verification regardless
                    # of its other quality scores
                    is_better_quality = quality_data["quality"] > self.best_quality
                    face_upgrade = quality_data.get("has_face") and not self._best_has_face
                    no_face_downgrade = (not quality_data.get("has_face")) and self._best_has_face

                    if (is_better_quality or face_upgrade) and not no_face_downgrade:
                        self.best_frame = frame.copy()
                        self.best_quality = quality_data["quality"]
                        self.best_size_check = quality_data.get("size_info")
                        self._best_has_face = quality_data.get("has_face", False)
                        self._best_face_bbox = quality_data.get("face_bbox")
                    
                    try:
                        self.frame_queue.put_nowait((frame, quality_data))
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait((frame, quality_data))
                        except:
                            pass
    
    def start_capture(self, delay_seconds=AUTO_CAPTURE_DELAY):
        """Start capture process. The live preview keeps tracking the best-quality
        frame seen so far, but the actual photo is only taken when the user presses
        ENTER - this lets the user wait until the card/face are clearly visible
        before locking in the image used for verification."""
        print(f"\n  📸 Hold NIC in front of camera — press ENTER to capture, ESC to cancel")

        if not self.face_detector.is_available():
            print_warn("No face detection backend available - install 'mediapipe' for best results "
                       "(pip install mediapipe). Falling back to a less accurate detector.")

        self.camera, cam_index, cam_backend = open_webcam()
        if self.camera is None:
            print_camera_troubleshooting()
            return {"success": False, "error": "Could not open camera - see troubleshooting steps above"}
        print(f"  🎥 Camera ready (index {cam_index}, backend {cam_backend})")
        
        # Set camera properties for better quality
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        self.is_capturing = True
        self.capture_complete = False
        self.best_frame = None
        self.best_quality = 0
        self.frames_analyzed = 0
        self.frames_with_nic = 0
        self.frames_with_face = 0
        self.quality_history = []
        self._best_has_face = False
        self._best_face_bbox = None
        
        capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        capture_thread.start()
        
        start_time = time.time()
        
        # Show preview with enhanced quality indicators
        while not self.capture_complete:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            elapsed = time.time() - start_time
            
            display = frame.copy()
            h, w = display.shape[:2]
            
            # Draw NIC guide with more detail
            card_w = int(w * 0.35)
            card_h = int(card_w / NIC_ASPECT_RATIO)
            x = (w - card_w) // 2
            y = (h - card_h) // 2
            
            # Draw guide with different colors based on quality
            current_quality = self._calculate_frame_quality(frame)
            guide_color = (0, 255, 0)  # Green default
            
            if current_quality.get("has_nic") and current_quality.get("size_valid"):
                guide_color = (0, 255, 0)  # Green - Good
            elif current_quality.get("has_nic"):
                guide_color = (0, 255, 255)  # Yellow - NIC detected but size issue
            else:
                guide_color = (0, 0, 255)  # Red - No NIC
            
            cv2.rectangle(display, (x, y), (x + card_w, y + card_h), guide_color, 3)
            
            # Labels
            cv2.putText(display, "Place NIC Here", (x, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide_color, 2)
            
            # Draw detected face bounding box, if any
            face_bbox = current_quality.get("face_bbox")
            if face_bbox:
                # Force plain Python ints - OpenCV 5.x + NumPy 2.x can reject
                # NumPy integer scalars (e.g. numpy.int64) here even though
                # older builds tolerated them, causing a "wrong type" error.
                fx, fy, fw, fh = (int(v) for v in face_bbox)
                cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (255, 128, 0), 2)
                cv2.putText(display, "Face", (fx, max(0, fy - 8)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
            
            # Elapsed time - LARGER AND MORE VISIBLE (no forced countdown anymore;
            # capture only happens when the user presses ENTER)
            cv2.putText(display, f"⏱️ {int(elapsed)}s", 
                       (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Quality bar
            if self.best_quality > 0:
                bar_x, bar_y = 10, 50
                bar_w = 250
                bar_h = 20
                fill_w = int(self.best_quality * bar_w)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
                cv2.putText(display, f"Best Quality: {self.best_quality:.0%}", 
                           (bar_x, bar_y + bar_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # NIC status
            if current_quality.get("has_nic"):
                cv2.putText(display, "✅ NIC Detected", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if current_quality.get("size_valid"):
                    cv2.putText(display, "✅ Proper Size", (10, 160), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display, "⚠️ Adjust Size/Position", (10, 160), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(display, "❌ No NIC Detected", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Face status
            if current_quality.get("has_face"):
                cv2.putText(display, "✅ Face Detected", (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "❌ No Face Detected", (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Glare / brightness warning
            glare_info = current_quality.get("glare_info", {})
            if glare_info.get("has_glare"):
                cv2.putText(display, "⚠️ Glare on card - tilt slightly", (10, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            elif glare_info.get("too_dark"):
                cv2.putText(display, "⚠️ Too dark - add more light", (10, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            elif glare_info.get("too_bright"):
                cv2.putText(display, "⚠️ Overexposed - reduce light", (10, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
            
            # Frame count and info
            cv2.putText(display, f"Frames: {self.frames_analyzed} | NIC: {self.frames_with_nic} | Face: {self.frames_with_face}", 
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Instructions
            status_text = "Press ENTER to capture this frame  |  ESC to cancel"
            cv2.putText(display, status_text, 
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("NIC Verification - Press ENTER to capture", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to cancel
                self.capture_complete = True
                break
            
            if key in (13, 10):  # ENTER pressed - capture the frame on screen right now
                self.best_frame = frame.copy()
                self.best_quality = current_quality["quality"]
                self.best_size_check = current_quality.get("size_info")
                self._best_has_face = current_quality.get("has_face", False)
                self._best_face_bbox = current_quality.get("face_bbox")
                self.capture_complete = True
                break
        
        self.is_capturing = False
        
        # Evaluate captured frame
        if self.capture_complete and self.best_frame is not None:
            # Final quality check
            final_quality = self._calculate_frame_quality(self.best_frame)
            size_check = check_nic_size_and_quality(self.best_frame)
            
            # Crop NIC region if detected
            if final_quality.get("has_nic"):
                self.captured_image = crop_nic_region(self.best_frame)
            else:
                self.captured_image = self.best_frame
            
            self.quality_score = self.best_quality
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"[memory] auto_capture_{timestamp}"
            
            result = {
                "success": True,
                "image": self.captured_image,
                "image_path": image_path,
                "quality": self.best_quality,
                "frames_analyzed": self.frames_analyzed,
                "frames_with_nic": self.frames_with_nic,
                "frames_with_face": self.frames_with_face,
                "nic_detected": final_quality.get("has_nic", False),
                "face_detected": final_quality.get("has_face", False),
                "face_confidence": final_quality.get("face_confidence", 0),
                "glare_info": final_quality.get("glare_info", {}),
                "size_valid": size_check.get("valid", False) if size_check else False,
                "size_info": size_check if size_check else None,
                "capture_time": round(time.time() - start_time, 1)
            }
        else:
            result = {"success": False, "error": "No valid image captured"}
        
        if self.camera is not None:
            self.camera.release()
        cv2.destroyAllWindows()
        
        return result

# =============================================================================
# ENHANCED VERIFICATION FUNCTIONS WITH AI
# =============================================================================

def match_images_with_ai(image1, image2, use_ai=True):
    """Match two images using AI face recognition and image comparison"""
    result = {
        "overall_score": 0,
        "status": "ERROR",
        "message": "Comparison failed",
        "face_match": False,
        "face_similarity": 0,
        "image_match": 0
    }
    
    try:
        if image1 is None or image2 is None:
            result["message"] = "Missing image data"
            return result
        
        overall_score, method_scores = compare_images_combined(image1, image2)
        result["image_match"] = overall_score
        result["method_scores"] = method_scores
        
        face_verified = False
        if use_ai and AI_FACE_AVAILABLE:
            agent = get_face_agent()
            if agent.is_available():
                face_result = agent.verify_face_match(image1, image2, FACE_MATCH_THRESHOLD)
                
                if face_result.get('success', False):
                    face_sim = face_result.get('similarity', 0)
                    face_match = face_result.get('match', False)
                    
                    result["face_similarity"] = face_sim
                    result["face_match"] = face_match
                    result["face_model"] = face_result.get('model', 'unknown')
                    
                    if face_sim >= 70:
                        overall_score = max(overall_score, face_sim * 0.9 + 10)
                        result["overall_score"] = overall_score
                        face_verified = True
                else:
                    result["face_error"] = face_result.get('error', 'Unknown error')
        
        if face_verified or overall_score >= 75:
            result["status"] = "APPROVED"
            result["message"] = f"Match confirmed ({overall_score:.1f}%)"
            if result.get("face_match"):
                result["message"] += " | Face: ✅"
        elif overall_score >= 60:
            result["status"] = "REVIEW"
            result["message"] = f"Partial match - needs review ({overall_score:.1f}%)"
            if result.get("face_match", False):
                result["message"] += " | Face: ✅"
            else:
                result["message"] += " | Face: ⚠️"
        elif overall_score >= 40:
            result["status"] = "REVIEW"
            result["message"] = f"Low confidence match - review required ({overall_score:.1f}%)"
        else:
            result["status"] = "REJECTED"
            result["message"] = f"No match detected ({overall_score:.1f}%)"
        
        result["overall_score"] = overall_score
        
    except Exception as e:
        result["message"] = f"Error: {str(e)[:50]}"
    
    return result

def verify_captured_with_prd(captured_image, nic_number):
    """Verify captured image with PRD records"""
    try:
        clean_nic = clean_nic_for_api(nic_number)
        if not is_valid_nic_format(clean_nic):
            return {"success": False, "error": f"Invalid NIC number '{nic_number}' (must be 9 digits followed by V/v, or 12 digits)"}
        
        prd_response = query_prd_customer(clean_nic, nic_front_image=encode_image_to_base64(captured_image))
        prd_data = extract_prd_data(prd_response)
        
        if not prd_data:
            return {"success": False, "error": "No PRD data found"}
        
        print_status("  PRD NIC Number", prd_data.get("nic", "N/A"))
        
        if not prd_data.get('nic_front_image'):
            return {"success": False, "error": "No NIC image found in PRD"}
        
        prd_image = decode_base64_image(prd_data['nic_front_image'])
        comparison = match_images_with_ai(captured_image, prd_image, use_ai=True)
        
        result = {
            "success": True,
            "nic": clean_nic,
            "prd_data": prd_data,
            "comparison": comparison,
            "status": comparison.get("status", "ERROR"),
            "score": comparison.get("overall_score", 0),
            "face_match": comparison.get("face_match", False),
            "face_similarity": comparison.get("face_similarity", 0)
        }
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def verify_stored_image_with_prd(image_path, nic_number):
    """Verify a stored image with PRD records"""
    try:
        if not os.path.exists(image_path):
            return {"success": False, "error": "Image file not found"}
        
        image = load_local_image(image_path)
        return verify_captured_with_prd(image, nic_number)
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def verify_stored_image_with_lime_and_prd(image_path, nic_number, mobile_number=None):
    """Verify a stored image with LIME and PRD.

    NIC is the required identifier (used both for the LIME lookup and the
    PRD query). Mobile is optional and, if given, only cross-checked
    against whatever mobile number is found in LIME's own metadata for
    the matched document - it is never used to search LIME.
    """
    try:
        if not os.path.exists(image_path):
            return {"success": False, "error": "Image file not found"}
        
        image = load_local_image(image_path)

        clean_nic = clean_nic_for_api(nic_number) if nic_number else None
        if not is_valid_nic_format(clean_nic):
            return {"success": False, "error": f"Invalid or missing NIC number '{nic_number}' (must be 9 digits followed by V/v, or 12 digits)"}

        if mobile_number and not validate_mobile_number(mobile_number):
            return {"success": False, "error": "Invalid mobile number format"}
        
        lime_result = find_lime_documents_and_metadata(nic_number, mobile_number)
        lime_front_doc = lime_result["front_doc"]
        lime_meta = lime_result["lime_meta"]
        lime_nic = lime_result["lime_nic"]
        lime_image = None

        if lime_front_doc:
            try:
                download_key = lime_front_doc.get("DownloadKey")
                if download_key:
                    download_response = download_lime_document(download_key)
                    lime_image_base64 = extract_image_data(download_response)
                    lime_image = decode_base64_image(lime_image_base64)
            except Exception as e:
                log_message(f"LIME image download failed: {e}", "WARNING")
        
        nic_for_prd, nic_source, nic_error = resolve_nic_for_prd(lime_nic, nic_number)
        if nic_error:
            return {"success": False, "error": nic_error}
        prd_response = query_prd_customer(nic_for_prd, nic_front_image=encode_image_to_base64(image))
        prd_data = extract_prd_data(prd_response)
        
        if not prd_data:
            return {"success": False, "error": "No PRD data found"}
        
        print_status("  PRD NIC Number", prd_data.get("nic", "N/A"))
        
        prd_image = None
        if prd_data.get('nic_front_image'):
            prd_image = decode_base64_image(prd_data['nic_front_image'])
        
        stored_prd_comparison = None
        if prd_image is not None:
            stored_prd_comparison = match_images_with_ai(image, prd_image, use_ai=True)
        
        lime_prd_comparison = None
        if lime_image is not None and prd_image is not None:
            lime_prd_comparison = match_images_with_ai(lime_image, prd_image, use_ai=True)
        
        stored_lime_comparison = None
        if lime_image is not None:
            stored_lime_comparison = match_images_with_ai(image, lime_image, use_ai=True)
        
        final_status = "REJECTED"
        final_score = 0
        final_message = "No match found"
        
        if stored_prd_comparison:
            score = stored_prd_comparison.get("overall_score", 0)
            if stored_prd_comparison.get("face_match", False):
                score += 10
                score = min(100, score)
            
            final_score = score
            
            if score >= 75:
                final_status = "APPROVED"
                final_message = "Strong match with PRD"
            elif score >= 60:
                final_status = "REVIEW"
                final_message = "Partial match with PRD"
            else:
                final_status = "REJECTED"
                final_message = "No match with PRD"
        
        if final_status in ["REJECTED", "REVIEW"] and stored_lime_comparison:
            lime_score = stored_lime_comparison.get("overall_score", 0)
            if lime_score > final_score:
                final_score = lime_score
                if lime_score >= 75:
                    final_status = "APPROVED"
                    final_message = "Match with LIME document"
                elif lime_score >= 60:
                    final_status = "REVIEW"
                    final_message = "Partial match with LIME"
        
        result = {
            "success": True,
            "nic_searched": clean_nic,
            "mobile_provided": lime_result["mobile_provided"],
            "lime_mobile": lime_result["lime_mobile"],
            "mobile_match": lime_result["mobile_match"],
            "lime_found": lime_nic is not None,
            "lime_nic": lime_nic,
            "lime_name": lime_meta.get("name"),
            "lime_address": lime_meta.get("address"),
            "lime_metadata_debug": lime_result["attempts"],
            "lime_matched_via": lime_result["matched_via"],
            "prd_data": prd_data,
            "final_status": final_status,
            "final_score": final_score,
            "final_message": final_message,
            "stored_prd_comparison": stored_prd_comparison,
            "lime_prd_comparison": lime_prd_comparison,
            "stored_lime_comparison": stored_lime_comparison,
            "face_match": stored_prd_comparison.get("face_match", False) if stored_prd_comparison else False,
            "face_similarity": stored_prd_comparison.get("face_similarity", 0) if stored_prd_comparison else 0,
            "image_path": image_path
        }
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def verify_with_lime_and_prd(captured_image, mobile_number, nic_number=None):
    """Verify captured image with LIME and PRD"""
    try:
        if not validate_mobile_number(mobile_number):
            return {"success": False, "error": "Invalid mobile number format"}
        
        lime_result = find_lime_documents_via_mobile(mobile_number, nic_number)
        lime_front_doc = lime_result["front_doc"]
        lime_meta = lime_result["lime_meta"]
        lime_nic = lime_result["lime_nic"]
        formatted_mobile = lime_result["formatted_mobile"]
        lime_image = None

        if lime_front_doc:
            try:
                download_key = lime_front_doc.get("DownloadKey")
                if download_key:
                    download_response = download_lime_document(download_key)
                    lime_image_base64 = extract_image_data(download_response)
                    lime_image = decode_base64_image(lime_image_base64)
            except Exception as e:
                log_message(f"LIME image download failed: {e}", "WARNING")
        
        nic_for_prd, nic_source, nic_error = resolve_nic_for_prd(lime_nic, nic_number)
        if nic_error:
            return {"success": False, "error": nic_error}
        prd_response = query_prd_customer(nic_for_prd, nic_front_image=encode_image_to_base64(captured_image))
        prd_data = extract_prd_data(prd_response)
        
        if not prd_data:
            return {"success": False, "error": "No PRD data found"}
        
        print_status("  PRD NIC Number", prd_data.get("nic", "N/A"))
        
        prd_image = None
        if prd_data.get('nic_front_image'):
            prd_image = decode_base64_image(prd_data['nic_front_image'])
        
        capture_prd_comparison = None
        if prd_image is not None:
            capture_prd_comparison = match_images_with_ai(captured_image, prd_image, use_ai=True)
        
        lime_prd_comparison = None
        if lime_image is not None and prd_image is not None:
            lime_prd_comparison = match_images_with_ai(lime_image, prd_image, use_ai=True)
        
        capture_lime_comparison = None
        if lime_image is not None:
            capture_lime_comparison = match_images_with_ai(captured_image, lime_image, use_ai=True)
        
        final_status = "REJECTED"
        final_score = 0
        final_message = "No match found"
        
        if capture_prd_comparison:
            score = capture_prd_comparison.get("overall_score", 0)
            if capture_prd_comparison.get("face_match", False):
                score += 10
                score = min(100, score)
            
            final_score = score
            
            if score >= 75:
                final_status = "APPROVED"
                final_message = "Strong match with PRD"
            elif score >= 60:
                final_status = "REVIEW"
                final_message = "Partial match with PRD"
            else:
                final_status = "REJECTED"
                final_message = "No match with PRD"
        
        if final_status in ["REJECTED", "REVIEW"] and capture_lime_comparison:
            lime_score = capture_lime_comparison.get("overall_score", 0)
            if lime_score > final_score:
                final_score = lime_score
                if lime_score >= 75:
                    final_status = "APPROVED"
                    final_message = "Match with LIME document"
                elif lime_score >= 60:
                    final_status = "REVIEW"
                    final_message = "Partial match with LIME"
        
        result = {
            "success": True,
            "mobile": formatted_mobile,
            "lime_found": lime_nic is not None,
            "lime_nic": lime_nic,
            "lime_name": lime_meta.get("name"),
            "lime_address": lime_meta.get("address"),
            "lime_metadata_debug": lime_result["attempts"],
            "lime_matched_via": lime_result["matched_via"],
            "prd_data": prd_data,
            "final_status": final_status,
            "final_score": final_score,
            "final_message": final_message,
            "capture_prd_comparison": capture_prd_comparison,
            "lime_prd_comparison": lime_prd_comparison,
            "capture_lime_comparison": capture_lime_comparison,
            "face_match": capture_prd_comparison.get("face_match", False) if capture_prd_comparison else False,
            "face_similarity": capture_prd_comparison.get("face_similarity", 0) if capture_prd_comparison else 0
        }
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def verify_lime_repository_with_prd(nic_number, mobile_number=None):
    """
    Verify a customer using ONLY the LIME repository and PRD - no camera
    capture, no locally stored image is used or written to disk.

    NIC is the required search key (mobile number is optional - it is
    never used to search LIME, only cross-checked against whatever mobile
    number is found in the matched document's own metadata, if any).

    Flow:
      1. Search LIME by NIC, pull the KYC_Front / NIC_Front document
         (per LIME_DOC_TYPES) and its MetaData (name, NIC, address, mobile).
      2. Download that document's image straight from LIME into memory.
      3. Query PRD for the same NIC and pull the PRD NIC image + data.
      4. Compare the LIME image against the PRD image (face + image match).
      5. Compare LIME metadata (name, NIC, address) against PRD data.
      6. If a mobile number was given, cross-check it against LIME's own
         mobile metadata for this document (informational, non-blocking).
      7. Combine into a single decision.
    """
    try:
        clean_nic = clean_nic_for_api(nic_number) if nic_number else None
        if not is_valid_nic_format(clean_nic):
            return {"success": False, "error": f"Invalid or missing NIC number '{nic_number}' (must be 9 digits followed by V/v, or 12 digits)"}

        if mobile_number and not validate_mobile_number(mobile_number):
            return {"success": False, "error": "Invalid mobile number format"}

        # ---- 1. Find the NIC/KYC front document in LIME, by NIC --------
        lime_result = find_lime_documents_and_metadata(nic_number, mobile_number)
        front_doc = lime_result["front_doc"]
        lime_meta = lime_result["lime_meta"]
        lime_nic = lime_result["lime_nic"]

        if not front_doc:
            return {
                "success": False,
                "error": "No KYC_Front / NIC_Front document found in LIME repository for this NIC. "
                         + format_lime_search_diagnostics(lime_result)
            }

        download_key = front_doc.get("DownloadKey")
        if not download_key:
            return {"success": False, "error": "LIME document has no DownloadKey"}

        download_response = download_lime_document(download_key)
        lime_image_base64 = extract_image_data(download_response)
        lime_image = decode_base64_image(lime_image_base64)
        if lime_image is None:
            return {"success": False, "error": "Could not decode LIME repository image"}

        lime_doc_type = front_doc.get("DocType", "N/A")

        # ---- 2. Query PRD for the matching customer ---------------------
        nic_for_prd, nic_source, nic_error = resolve_nic_for_prd(lime_nic, nic_number)
        if nic_error:
            return {"success": False, "error": nic_error}

        prd_response = query_prd_customer(nic_for_prd, nic_front_image=lime_image_base64)
        prd_data = extract_prd_data(prd_response)
        if not prd_data:
            return {"success": False, "error": "No PRD data found"}

        print_status("  PRD NIC Number", prd_data.get("nic", "N/A"))

        if not prd_data.get('nic_front_image'):
            return {"success": False, "error": "PRD record has no NIC front image to compare against"}

        prd_image = decode_base64_image(prd_data['nic_front_image'])

        # ---- 3. Image comparison: LIME repository image vs PRD image ---
        lime_prd_comparison = match_images_with_ai(lime_image, prd_image, use_ai=True)
        image_score = lime_prd_comparison.get("overall_score", 0)

        # ---- 4. Data comparison: LIME metadata vs PRD data --------------
        data_comparison = compare_customer_data(lime_meta, prd_data, lime_nic_normalized=lime_nic)
        data_score = data_comparison.get("overall_score", 0)

        # ---- 5. Combine into a final decision ---------------------------
        # Image match carries more weight since it's the core identity check;
        # the data-field match acts as a corroborating/blocking signal.
        if data_comparison["fields"]["nic"]["lime_value"] and data_comparison["fields"]["nic"]["prd_value"]:
            final_score = (image_score * 0.7) + (data_score * 0.3)
        else:
            # No NIC to cross-check - rely on the image match alone
            final_score = image_score

        if lime_prd_comparison.get("face_match", False):
            final_score += 10
            final_score = min(100, final_score)

        if final_score >= 75 and data_comparison["all_match"]:
            final_status = "APPROVED"
            final_message = "LIME repository image and data match PRD"
        elif final_score >= 75:
            final_status = "REVIEW"
            final_message = "Image matches PRD but some customer data differs - review required"
        elif final_score >= 60:
            final_status = "REVIEW"
            final_message = "Partial match between LIME repository and PRD"
        else:
            final_status = "REJECTED"
            final_message = "No match between LIME repository and PRD"

        # Mobile is informational only here (not part of the search or the
        # score) - note it if a mismatch was found against LIME's own data.
        if lime_result["mobile_match"] is False:
            final_message += (
                f" (note: provided mobile {mobile_number} does not match "
                f"LIME's mobile for this NIC, {lime_result['lime_mobile']})"
            )

        result = {
            "success": True,
            "nic_searched": clean_nic,
            "mobile_provided": lime_result["mobile_provided"],
            "lime_mobile": lime_result["lime_mobile"],
            "mobile_match": lime_result["mobile_match"],
            "lime_found": True,
            "lime_nic": lime_nic,
            "lime_name": lime_meta.get("name"),
            "lime_address": lime_meta.get("address"),
            "lime_metadata_debug": lime_result["attempts"],
            "lime_matched_via": lime_result["matched_via"],
            "lime_doc_type": lime_doc_type,
            "prd_data": prd_data,
            "final_status": final_status,
            "final_score": final_score,
            "final_message": final_message,
            "lime_prd_comparison": lime_prd_comparison,
            "data_comparison": data_comparison,
            "face_match": lime_prd_comparison.get("face_match", False),
            "face_similarity": lime_prd_comparison.get("face_similarity", 0)
        }

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# SIGNATURE VERIFICATION - local matcher (v2)
#
# Base64.ai (verify_signature_with_base64ai below) is still the primary
# decision when configured and conclusive; this local matcher is computed
# on every request and decides whenever Base64.ai is off, down or
# inconclusive.
#
# How it works (v2):
#   1. Ink extraction - adaptive threshold, then every candidate stroke must
#      be (a) dark enough to be pen ink (drops light watermarks) and (b)
#      clearly darker than the paper on BOTH sides of it (drops paper/table
#      edges and shadow boundaries in phone photos).
#   2. Normalisation - ink is thinned to a 1px skeleton and turned into a
#      point cloud centred on its centre of mass and scaled to a fixed
#      "radius of gyration". This removes image size, crop, margins, pen
#      colour and pen thickness. Unlike the old bounding-box fit, it is not
#      thrown off when one signature is taller/wider than the other.
#   3. Alignment - a coarse search over rotation (+/-60 deg), mild
#      horizontal/vertical stretch and slant, then a fine affine fit that
#      minimises the two-way chamfer distance between the strokes. Stretch
#      and slant are penalised, so the fit can absorb normal signing
#      variation and a tilted photo but cannot freely bend an unrelated
#      signature into shape.
#   4. Scoring on the aligned strokes:
#        - closeness:     average distance between the two stroke sets
#        - coverage:      share of each signature's strokes that have a
#                         stroke from the other one nearby (both ways)
#        - shape_context: local structure around each point (which way the
#                         strokes run and where the rest of the signature
#                         sits) compared at corresponding positions
#      overall = 0.40*closeness + 0.30*coverage + 0.30*shape_context
#
# Thresholds were set from the sample genuine pair (scanned specimen vs
# phone photo, ~62) plus synthetic genuine variants and synthetic
# impostors. They are a starting point: run the CLI below on a few dozen
# real genuine/non-matching pairs from LIME and adjust via the env vars
# SIGNATURE_APPROVE_THRESHOLD / SIGNATURE_REVIEW_THRESHOLD.
#
#   python NIC-Sig-NoENV-NF.py --compare reference.png captured.png
# =============================================================================

SIG_MAX_WORKING_DIM = 1000      # downscale huge phone photos before thresholding
SIG_NORM_CANVAS = 512           # working canvas for the normalised strokes
SIG_NORM_RADIUS = 90.0          # radius of gyration the ink is scaled to
SIG_MAX_ROTATION_DEG = 60       # widest tilt between the two captures we search
SIG_DEFORM_PENALTY = 8.0        # cost per unit of stretch/slant in the fit
SIG_COVERAGE_TOL_PX = 6         # "nearby" for coverage, in normalised pixels
SIG_MIN_SKELETON_POINTS = 40    # less ink than this = no usable signature
_SIG_FIT_BOUNDS = [
    (-np.deg2rad(SIG_MAX_ROTATION_DEG + 5), np.deg2rad(SIG_MAX_ROTATION_DEG + 5)),
    (np.log(0.6), np.log(1.6)),   # log x-scale
    (np.log(0.6), np.log(1.6)),   # log y-scale
    (-0.6, 0.6),                  # slant (shear)
    (-60, 60), (-60, 60),         # translation
]


class NoSignatureInkError(ImageComparisonError):
    pass


def _signature_ink_mask(image, ink_darkness_cutoff=180, min_area=10,
                        min_contrast=25, border_contrast=40):
    """Binary mask of real pen ink.

    Candidate strokes come from an adaptive threshold (works under uneven
    lighting). A candidate is kept only if it is dark enough to be ink and
    clearly darker than the paper 4-10px around it. A paper/desk edge or a
    shadow line is only darker than one side, so its surroundings are about
    as dark as it is and it is dropped. Components touching the image border
    must pass a stricter contrast check, since that is where paper edges live.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    h, w = gray.shape[:2]
    scale = SIG_MAX_WORKING_DIM / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 10)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ink = np.zeros_like(mask)
    H, W = gray.shape
    pad = 12
    k_inner, k_outer = np.ones((9, 9), np.uint8), np.ones((21, 21), np.uint8)
    for lbl in range(1, num_labels):
        x, y, cw, ch, area = stats[lbl]
        if area < min_area or (cw < 3 and ch < 3):
            continue
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + cw + pad), min(H, y + ch + pad)
        comp = (labels[y0:y1, x0:x1] == lbl).astype(np.uint8)
        g = gray[y0:y1, x0:x1]
        ink_level = float(g[comp > 0].mean())
        if ink_level > ink_darkness_cutoff:
            continue  # too light to be pen ink (watermark, faint print)
        ring = ((cv2.dilate(comp, k_outer) > 0) & (cv2.dilate(comp, k_inner) == 0)
                & (mask[y0:y1, x0:x1] == 0))
        if ring.sum() >= 10:
            contrast = float(np.median(g[ring])) - ink_level
            touches_border = x <= 1 or y <= 1 or x + cw >= W - 1 or y + ch >= H - 1
            if contrast < (border_contrast if touches_border else min_contrast):
                continue
        ink[y0:y1, x0:x1][comp > 0] = 255
    return ink


def _signature_points(image):
    """Ink -> 1px skeleton -> point cloud centred at the origin and scaled to
    a fixed radius of gyration."""
    mask = _signature_ink_mask(image)
    ys, xs = np.nonzero(mask)
    if len(xs) < SIG_MIN_SKELETON_POINTS:
        raise NoSignatureInkError("No signature ink could be detected in the image")

    # Rescale the mask first so the skeleton is taken at a consistent
    # resolution regardless of the original image size.
    rg = np.sqrt(((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2).mean())
    s = SIG_NORM_RADIUS / max(rg, 1e-6)
    h, w = mask.shape
    scaled = cv2.resize(mask, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(skeletonize(scaled > 127))
    if len(xs) < SIG_MIN_SKELETON_POINTS:
        raise NoSignatureInkError("No signature ink could be detected in the image")

    pts = np.stack([xs, ys], 1).astype(np.float64)
    pts -= pts.mean(0)
    pts *= SIG_NORM_RADIUS / np.sqrt((pts ** 2).sum(1).mean())
    return pts


def _render_points(pts, size=SIG_NORM_CANVAS):
    img = np.zeros((size, size), np.uint8)
    p = np.round(pts + size / 2).astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 0] < size) & (p[:, 1] >= 0) & (p[:, 1] < size)
    img[p[ok, 1], p[ok, 0]] = 255
    return img


def _distance_field(pts):
    return ndimage.distance_transform_edt(_render_points(pts) == 0)


def _sample_field(field, pts):
    p = pts + SIG_NORM_CANVAS / 2
    return ndimage.map_coordinates(field, [p[:, 1], p[:, 0]], order=1, mode="nearest")


def _affine(params):
    theta, log_sx, log_sy, shear, tx, ty = params
    c, s = np.cos(theta), np.sin(theta)
    A = np.array([[c, -s], [s, c]]) @ np.array([[np.exp(log_sx), shear], [0.0, np.exp(log_sy)]])
    return A, np.array([tx, ty])


def _deformation(params):
    """How far a fit is from pure rotation+shift (stretch + slant)."""
    return abs(params[1]) + abs(params[2]) + abs(params[3])


def _subsample(pts, n, seed=0):
    if len(pts) <= n:
        return pts
    return pts[np.random.default_rng(seed).choice(len(pts), n, replace=False)]


def _align_signatures(ref_pts, qry_pts):
    """Find the affine transform mapping the query strokes onto the reference
    strokes (two-way chamfer, with stretch/slant penalised)."""
    ref_field, qry_field = _distance_field(ref_pts), _distance_field(qry_pts)

    def make_cost(q, r):
        def cost(params):
            A, t = _affine(params)
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                return 1e9
            d_qr = np.minimum(_sample_field(ref_field, q @ A.T + t), 25).mean()
            d_rq = np.minimum(_sample_field(qry_field, (r - t) @ A_inv.T), 25).mean()
            return (d_qr + d_rq) / 2 + SIG_DEFORM_PENALTY * _deformation(params)
        return cost

    coarse_cost = make_cost(_subsample(qry_pts, 200, 1), _subsample(ref_pts, 200, 2))
    fine_cost = make_cost(_subsample(qry_pts, 500, 3), _subsample(ref_pts, 500, 4))

    candidates = []
    for theta in np.deg2rad(np.arange(-SIG_MAX_ROTATION_DEG, SIG_MAX_ROTATION_DEG + 1, 5)):
        for lsx in np.log([0.8, 1.0, 1.25]):
            for lsy in np.log([0.8, 1.0, 1.25]):
                for shear in (-0.2, 0.0, 0.2):
                    p = np.array([theta, lsx, lsy, shear, 0.0, 0.0])
                    candidates.append((coarse_cost(p), p))
    candidates.sort(key=lambda c: c[0])

    seeds = []
    for _, p in candidates:
        if all(abs(p[0] - s[0]) > np.deg2rad(12) for s in seeds):
            seeds.append(p)
        if len(seeds) == 4:
            break

    best = None
    for p0 in seeds:
        res = minimize(fine_cost, p0, method="Powell", bounds=_SIG_FIT_BOUNDS,
                       options={"xtol": 1e-2, "ftol": 1e-3, "maxiter": 800})
        if best is None or res.fun < best[0]:
            best = (res.fun, res.x)
    return best[1]


def _shape_context(pts, centers, n_r=5, n_t=12, r_in=0.125, r_out=2.0):
    """Log-polar histogram of the stroke points around each centre."""
    d = pts[None, :, :] - centers[:, None, :]
    r = np.hypot(d[..., 0], d[..., 1]) / SIG_NORM_RADIUS
    t = np.mod(np.arctan2(d[..., 1], d[..., 0]), 2 * np.pi)
    r_bin = np.digitize(r, np.logspace(np.log10(r_in), np.log10(r_out), n_r + 1)) - 1
    t_bin = np.minimum((t / (2 * np.pi) * n_t).astype(int), n_t - 1)
    valid = (r_bin >= 0) & (r_bin < n_r)
    idx = r_bin * n_t + t_bin
    hist = np.zeros((len(centers), n_r * n_t))
    for i in range(len(centers)):
        hist[i] = np.bincount(idx[i][valid[i]], minlength=n_r * n_t)
    return hist / (hist.sum(1, keepdims=True) + 1e-9)


def _shape_context_score(a, b):
    """0-100 agreement of local structure between corresponding positions
    of two ALIGNED point clouds."""
    ca, cb = _subsample(a, 150, 5), _subsample(b, 150, 6)
    ha, hb = _shape_context(a, ca), _shape_context(b, cb)
    dist = np.linalg.norm(ca[:, None] - cb[None], axis=2)
    nn_ab, nn_ba = dist.argmin(1), dist.argmin(0)

    def chi2(h1, h2):
        return 0.5 * ((h1 - h2) ** 2 / (h1 + h2 + 1e-9)).sum(1)

    cost = (chi2(ha, hb[nn_ab]).mean() + chi2(hb, ha[nn_ba]).mean()) / 2
    # chi2 ~0.15 for the same signature, ~0.3-0.5 for different ones;
    # map 0.5 -> 0 and 0 -> 100.
    return float(np.clip((0.5 - cost) / 0.5 * 100, 0, 100))


def _overlay_image(ref_pts, aligned_pts):
    """PNG data URL showing reference (dark) vs aligned captured (blue)
    strokes - handy for a reviewer on REVIEW results."""
    ref = cv2.dilate(_render_points(ref_pts), np.ones((2, 2), np.uint8)) > 0
    qry = cv2.dilate(_render_points(aligned_pts), np.ones((2, 2), np.uint8)) > 0
    img = np.full((SIG_NORM_CANVAS, SIG_NORM_CANVAS, 3), 255, np.uint8)
    img[ref] = (40, 40, 40)
    img[qry] = (200, 90, 20)
    img[ref & qry] = (120, 40, 120)
    ys, xs = np.nonzero(ref | qry)
    if len(xs):
        y0, y1 = max(0, ys.min() - 10), min(SIG_NORM_CANVAS, ys.max() + 10)
        x0, x1 = max(0, xs.min() - 10), min(SIG_NORM_CANVAS, xs.max() + 10)
        img = img[y0:y1, x0:x1]
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


def compare_signature_images_combined(captured_image, reference_image):
    """Align the captured signature onto the reference and score them.

    Returns (overall, method_scores, hit_boundary, overlay_data_url).
    method_scores holds plain numbers only (the frontend breakdown expects
    that); hit_boundary is True when the best rotation sits at the edge of
    the search range.
    """
    ref = _signature_points(reference_image)
    qry = _signature_points(captured_image)

    params = _align_signatures(ref, qry)
    A, t = _affine(params)
    qry_aligned = qry @ A.T + t

    ref_field, qry_field = _distance_field(ref), _distance_field(qry_aligned)
    d_ref = _sample_field(qry_field, ref)          # reference strokes -> captured
    d_qry = _sample_field(ref_field, qry_aligned)  # captured strokes -> reference

    mean_dist = (np.minimum(d_ref, 30).mean() + np.minimum(d_qry, 30).mean()) / 2
    closeness = float(np.clip(100 * (1 - mean_dist / 15.0), 0, 100))

    cov_ref = (d_ref <= SIG_COVERAGE_TOL_PX).mean()
    cov_qry = (d_qry <= SIG_COVERAGE_TOL_PX).mean()
    # average of mean and worst direction: a signature that only matches a
    # part of the other one (missing/extra strokes) is pulled down.
    coverage = float(100 * ((cov_ref + cov_qry) / 2 + min(cov_ref, cov_qry)) / 2)

    shape_ctx = _shape_context_score(ref, qry_aligned)

    overall = 0.40 * closeness + 0.30 * coverage + 0.30 * shape_ctx

    rotation = float(np.rad2deg(params[0]))
    method_scores = {
        "closeness": round(closeness, 1),
        "coverage": round(coverage, 1),
        "shape_context": round(shape_ctx, 1),
        "rotation_degrees": round(rotation, 1),
        "stretch_slant": round(float(_deformation(params)), 2),
    }
    hit_boundary = abs(rotation) >= SIG_MAX_ROTATION_DEG
    return float(overall), method_scores, hit_boundary, _overlay_image(ref, qry_aligned)


def verify_signature_with_base64ai(reference_image, query_image, settings=None):
    """Call Base64.ai's Signature AI (/api/signature) to verify whether two
    signature images match. This is used ONLY by compare_signatures() below
    for the signature-verification flow - it is never called from anywhere
    in the NIC/PRD verification code.

    reference_image is the ground-truth signature (Base64.ai's 'document'
    param - the LIME repository copy here), query_image is the one being
    validated against it (Base64.ai's 'query' param - the freshly captured
    or uploaded signature here). See:
    https://apidoc.base64.ai (section 4, "Signature recognition & verification").

    Returns a dict that never raises - callers check "available"/"error"
    instead of catching exceptions:
      {
        "available": bool,        # True if Base64.ai actually responded
        "matched": True/False/None,  # None = inconclusive (e.g. no
                                      # signature detected in one image) or
                                      # not available
        "matches": [...],         # raw match array from the API
        "is_reference_valid": ... # per Base64.ai docs, see is_reference_valid semantics above
        "message": "...",
        "error": None or str,
      }
    """
    result = {
        "available": False,
        "matched": None,
        "matches": [],
        "is_reference_valid": None,
        "message": "",
        "error": None,
    }

    if not BASE64AI_ENABLED:
        result["error"] = "Base64.ai is not configured (set BASE64AI_EMAIL / BASE64AI_API_KEY)"
        return result

    if reference_image is None or query_image is None:
        result["error"] = "Missing image data for Base64.ai signature verification"
        return result

    try:
        headers = {
            "Content-Type": "application/json",
            # Base64.ai auth format: "ApiKey <account email>:<secret>"
            "Authorization": f"ApiKey {BASE64AI_EMAIL}:{BASE64AI_API_KEY}",
        }
        payload = {
            "document": encode_image_to_base64_data_url(reference_image),
            "query": encode_image_to_base64_data_url(query_image),
            "settings": {
                "maxNumberOfSignatures": BASE64AI_MAX_SIGNATURES,
                "signatureDetectionThreshold": BASE64AI_DETECTION_THRESHOLD,
            },
        }
        if settings:
            payload["settings"].update(settings)

        response = requests.post(
            BASE64AI_SIGNATURE_URL, headers=headers, json=payload, timeout=BASE64AI_TIMEOUT_SECONDS
        )

        if not response.ok:
            try:
                body = response.json()
                detail = body.get("message") if isinstance(body, dict) else body
            except ValueError:
                detail = response.text[:500]
            result["error"] = f"Base64.ai rejected the request ({response.status_code}): {detail}"
            return result

        data = response.json()
        result["available"] = True

        # The documented success shape is a plain JSON array of matches.
        # Handled defensively in case a deployment wraps it in an object.
        if isinstance(data, list):
            matches, message, is_reference_valid = data, None, None
        elif isinstance(data, dict):
            matches = data.get("matches", [])
            message = data.get("message")
            is_reference_valid = data.get("isReferenceValid")
        else:
            matches, message, is_reference_valid = [], None, None

        result["matches"] = matches
        result["is_reference_valid"] = is_reference_valid

        if message:
            # Per the API docs, a message here means one or both images had
            # no detectable signature at all - inconclusive, not a mismatch.
            result["matched"] = None
            result["message"] = message
        else:
            result["matched"] = len(matches) > 0
            result["message"] = (
                "Base64.ai confirmed matching signatures"
                if result["matched"]
                else "Base64.ai found no matching signature"
            )

        return result

    except requests.exceptions.RequestException as e:
        result["error"] = f"Base64.ai request failed: {str(e)[:200]}"
        return result
    except Exception as e:
        result["error"] = f"Base64.ai signature verification error: {str(e)[:200]}"
        return result


def compare_signatures(image1, image2):
    """Compare a captured/uploaded signature (image1) with the stored LIME
    signature (image2).

    Decision source: Base64.ai's Signature AI (verify_signature_with_base64ai()
    above) is the primary decision when it's configured and gives a
    conclusive answer. The local matcher (compare_signature_images_combined())
    is always computed too and decides whenever Base64.ai is unconfigured,
    unreachable or inconclusive - so signature verification never depends
    on Base64.ai being up. Never runs the face-recognition agent.
    """
    result = {
        "overall_score": 0,
        "status": "ERROR",
        "message": "Comparison failed",
        "image_match": 0,
    }

    try:
        if image1 is None or image2 is None:
            result["message"] = "Missing signature image data"
            return result

        # ---- Local matcher ------------------------------------------------
        try:
            overall_score, method_scores, hit_boundary, overlay = compare_signature_images_combined(image1, image2)
            result["overall_score"] = round(overall_score, 1)
            result["image_match"] = round(overall_score, 1)
            result["method_scores"] = method_scores   # plain numbers only (frontend calls .toFixed)
            result["rotation_search_hit_boundary"] = hit_boundary
            result["overlay_image"] = overlay
            result["thresholds"] = {"approve": SIGNATURE_APPROVE_THRESHOLD,
                                    "review": SIGNATURE_REVIEW_THRESHOLD}

            if overall_score >= SIGNATURE_APPROVE_THRESHOLD and not hit_boundary:
                local_status = "APPROVED"
                local_message = f"Signature match confirmed ({overall_score:.1f}%)"
            elif overall_score >= SIGNATURE_REVIEW_THRESHOLD:
                local_status = "REVIEW"
                local_message = f"Partial signature match - needs manual review ({overall_score:.1f}%)"
                if hit_boundary:
                    local_message += " - signatures are rotated a lot relative to each other"
            else:
                local_status = "REJECTED"
                local_message = f"Signature does not match ({overall_score:.1f}%)"
        except NoSignatureInkError as e:
            overall_score = 0.0
            local_status = "ERROR"
            local_message = f"{e} - please recapture on plain paper with good lighting"

        result["status"] = local_status
        result["message"] = local_message

        # ---- Base64.ai Signature AI (signature verification only) --------
        # image1 = captured/uploaded ("query"), image2 = LIME ("document").
        b64ai_result = verify_signature_with_base64ai(reference_image=image2, query_image=image1)
        result["base64ai"] = b64ai_result

        if b64ai_result["available"] and b64ai_result["matched"] is not None:
            result["status"] = "APPROVED" if b64ai_result["matched"] else "REJECTED"
            result["message"] = f"{b64ai_result['message']} (local similarity {overall_score:.1f}%)"
        else:
            # Base64.ai unavailable, errored, or inconclusive - keep the
            # local decision computed above.
            # (no note when Base64.ai simply isn't configured - that's the
            # normal local-only setup, not something the operator must read)
            note = b64ai_result.get("error") or b64ai_result.get("message")
            if note and BASE64AI_ENABLED:
                result["message"] = f"{local_message} [Base64.ai: {note}]"

    except Exception as e:
        log_message(f"Signature comparison failed: {e}", "ERROR")
        result["message"] = f"Error: {str(e)[:120]}"

    return result


def _cli_compare(argv):
    """Offline calibration helper:
        python signature_verify_app.py --compare reference.png captured.png [more_captured...]
    Prints the score breakdown for each captured image against the reference
    and writes an alignment overlay next to it (<captured>_overlay.png)."""
    if len(argv) < 2:
        print("usage: python NIC-Sig-NoENV-NF.py --compare reference.png captured.png [...]")
        return 2
    missing = [p for p in argv if not os.path.isfile(p)]
    if missing:
        print(f"File(s) not found (current folder: {os.getcwd()}):")
        for p in missing:
            print(f"  - {p}")
        print("Use the real image file names, e.g.: --compare Original.png 1.png")
        if argv[0] in missing:
            return 2
    try:
        reference = load_local_image(argv[0])
    except Exception as e:
        print(f"Could not read reference image {argv[0]}: {e}")
        return 2
    for path in argv[1:]:
        if path in missing:
            continue
        try:
            captured = load_local_image(path)
        except Exception as e:
            print(f"{os.path.basename(path):30s} could not be read: {e}")
            continue
        res = compare_signatures(captured, reference)
        scores = res.get("method_scores", {})
        print(f"{os.path.basename(path):30s} {res['status']:9s} {res['overall_score']:5.1f}  "
              + "  ".join(f"{k}={v}" for k, v in scores.items()))
        if res.get("overlay_image"):
            out = os.path.splitext(path)[0] + "_overlay.png"
            with open(out, "wb") as f:
                f.write(base64.b64decode(res["overlay_image"].split(",", 1)[1]))
    return 0


def verify_signature_with_lime(nic_number, signature_image, mobile_number=None):
    """
    Verify a captured/uploaded customer signature against the SIGNATURE
    document stored in the LIME repository for that NIC.

    Flow:
      1. Search LIME by NIC, pull the SIGNATURE document and its MetaData.
      2. Download that document's image straight from LIME into memory.
      3. Compare the provided signature image against the LIME signature
         image using compare_signatures().
      4. If a mobile number was given, cross-check it against LIME's own
         mobile metadata for this document (informational, non-blocking).
    """
    try:
        clean_nic = clean_nic_for_api(nic_number) if nic_number else None
        if not is_valid_nic_format(clean_nic):
            return {"success": False, "error": f"Invalid or missing NIC number '{nic_number}' (must be 9 digits followed by V/v, or 12 digits)"}

        if mobile_number and not validate_mobile_number(mobile_number):
            return {"success": False, "error": "Invalid mobile number format"}

        if signature_image is None:
            return {"success": False, "error": "A captured or uploaded signature image is required"}

        # ---- 1. Find the SIGNATURE document in LIME, by NIC ---------------
        lime_result = find_lime_signature_document(nic_number, mobile_number)
        signature_doc = lime_result["signature_doc"]
        lime_meta = lime_result["lime_meta"]
        lime_nic = lime_result["lime_nic"]

        if not signature_doc:
            return {
                "success": False,
                "error": "No SIGNATURE document found in LIME repository for this NIC. "
                         + format_lime_signature_search_diagnostics(lime_result)
            }

        download_key = signature_doc.get("DownloadKey")
        if not download_key:
            return {"success": False, "error": "LIME signature document has no DownloadKey"}

        download_response = download_lime_document(download_key)
        lime_signature_base64 = extract_image_data(download_response)
        lime_signature_image = decode_base64_image(lime_signature_base64)
        if lime_signature_image is None:
            return {"success": False, "error": "Could not decode LIME repository signature image"}

        lime_doc_type = signature_doc.get("DocType", "N/A")

        # ---- 2. Compare captured/uploaded signature vs LIME signature -----
        signature_comparison = compare_signatures(signature_image, lime_signature_image)
        final_status = signature_comparison.get("status", "ERROR")
        final_score = signature_comparison.get("overall_score", 0)
        final_message = signature_comparison.get("message", "")

        if lime_result["mobile_match"] is False:
            final_message += (
                f" (note: provided mobile {mobile_number} does not match "
                f"LIME's mobile for this NIC, {lime_result['lime_mobile']})"
            )

        return {
            "success": True,
            "nic_searched": clean_nic,
            "mobile_provided": lime_result["mobile_provided"],
            "lime_mobile": lime_result["lime_mobile"],
            "mobile_match": lime_result["mobile_match"],
            "lime_found": True,
            "lime_nic": lime_nic,
            "lime_name": lime_meta.get("name"),
            "lime_metadata_debug": lime_result["attempts"],
            "lime_matched_via": lime_result["matched_via"],
            "lime_doc_type": lime_doc_type,
            "final_status": final_status,
            "final_score": final_score,
            "final_message": final_message,
            "signature_comparison": signature_comparison,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def verify_signature_with_lime(nic_number, signature_image, mobile_number=None):
    """
    Verify a captured/uploaded customer signature against the SIGNATURE
    document stored in the LIME repository for that NIC.

    This is the signature equivalent of verify_lime_repository_with_prd() /
    verify_stored_image_with_lime_and_prd() - same LIME-lookup pattern, same
    result shape, same APPROVED/REVIEW/REJECTED decision bands - but it
    compares against LIME's stored SIGNATURE doc instead of KYC_Front/
    NIC_Front, and it does NOT call PRD at all (PRD has no signature
    record) and does NOT run face recognition (no face in a signature).

    Flow:
      1. Search LIME by NIC, pull the SIGNATURE document (per
         LIME_SIGNATURE_DOC_TYPES) and its MetaData (name, NIC, mobile).
      2. Download that document's image straight from LIME into memory.
      3. Compare the provided signature image against the LIME signature
         image using compare_signatures() (Base64.ai if configured, else the
         local alignment-based signature matcher).
      4. If a mobile number was given, cross-check it against LIME's own
         mobile metadata for this document (informational, non-blocking).
    """
    try:
        clean_nic = clean_nic_for_api(nic_number) if nic_number else None
        if not is_valid_nic_format(clean_nic):
            return {"success": False, "error": f"Invalid or missing NIC number '{nic_number}' (must be 9 digits followed by V/v, or 12 digits)"}

        if mobile_number and not validate_mobile_number(mobile_number):
            return {"success": False, "error": "Invalid mobile number format"}

        if signature_image is None:
            return {"success": False, "error": "A captured or uploaded signature image is required"}

        # ---- 1. Find the SIGNATURE document in LIME, by NIC ---------------
        lime_result = find_lime_signature_document(nic_number, mobile_number)
        signature_doc = lime_result["signature_doc"]
        lime_meta = lime_result["lime_meta"]
        lime_nic = lime_result["lime_nic"]

        if not signature_doc:
            return {
                "success": False,
                "error": "No SIGNATURE document found in LIME repository for this NIC. "
                         + format_lime_signature_search_diagnostics(lime_result)
            }

        download_key = signature_doc.get("DownloadKey")
        if not download_key:
            return {"success": False, "error": "LIME signature document has no DownloadKey"}

        download_response = download_lime_document(download_key)
        lime_signature_base64 = extract_image_data(download_response)
        lime_signature_image = decode_base64_image(lime_signature_base64)
        if lime_signature_image is None:
            return {"success": False, "error": "Could not decode LIME repository signature image"}

        lime_doc_type = signature_doc.get("DocType", "N/A")

        # ---- 2. Compare captured/uploaded signature vs LIME signature -----
        signature_comparison = compare_signatures(signature_image, lime_signature_image)
        final_status = signature_comparison.get("status", "ERROR")
        final_score = signature_comparison.get("overall_score", 0)
        final_message = signature_comparison.get("message", "")

        # Mobile is informational only here (not part of the search or the
        # score) - note it if a mismatch was found against LIME's own data,
        # same convention as verify_lime_repository_with_prd().
        if lime_result["mobile_match"] is False:
            final_message += (
                f" (note: provided mobile {mobile_number} does not match "
                f"LIME's mobile for this NIC, {lime_result['lime_mobile']})"
            )

        result = {
            "success": True,
            "nic_searched": clean_nic,
            "mobile_provided": lime_result["mobile_provided"],
            "lime_mobile": lime_result["lime_mobile"],
            "mobile_match": lime_result["mobile_match"],
            "lime_found": True,
            "lime_nic": lime_nic,
            "lime_name": lime_meta.get("name"),
            "lime_metadata_debug": lime_result["attempts"],
            "lime_matched_via": lime_result["matched_via"],
            "lime_doc_type": lime_doc_type,
            "final_status": final_status,
            "final_score": final_score,
            "final_message": final_message,
            "signature_comparison": signature_comparison,
        }

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# SIMPLIFIED DISPLAY FUNCTIONS
# =============================================================================

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_error(message):
    """Print error message"""
    print(f"  ❌ {message}")

def print_warn(message):
    """Print warning message"""
    print(f"  ⚠️  {message}")

def print_status(label, value, status=None):
    """Print status line"""
    if status == "pass":
        print(f"  ✅ {label}: {value}")
    elif status == "fail":
        print(f"  ❌ {label}: {value}")
    elif status == "warn":
        print(f"  ⚠️  {label}: {value}")
    else:
        print(f"  {label}: {value}")

# =============================================================================
# WEB APPLICATION (Flask front-end, from app.py - merged, no external
# nic_verification_merged.py file needed; the core engine above IS this file)
# =============================================================================
"""
NIC Verification System - Web Application
==========================================
A Flask front-end for the verification engine defined above in this same
file, so specific users can reach it over the network from a browser
instead of running the old desktop Tkinter app locally.

Verification modes (all four preserved from the desktop app):
  1. Stored image + PRD            - upload a NIC photo, check against PRD
  2. Stored image + LIME + PRD     - upload a NIC photo, check against LIME
                                      and PRD
  3. LIME repository + PRD         - mobile/NIC number only, no photo upload
  4. Capture from camera + verify  - BROWSER camera (getUserMedia + canvas)
                                      captures a frame, which is uploaded as
                                      a normal photo for verification. (The
                                      desktop app read the SERVER's webcam,
                                      which makes no sense for a multi-user
                                      web app - this reads each visitor's own
                                      camera instead.)

ACCESS CONTROL
--------------
This app requires a login. Allowed users are configured via the
NIC_APP_USERS environment variable (see webapp.env.example / DEPLOY.md) as
"username:password_hash" pairs - never as plaintext passwords in code.
Use hash_password.py to generate a hash for each user you want to allow.

RUNNING
-------
Dev only:  python app.py
Production: gunicorn, behind nginx - see DEPLOY.md.
"""

# Offline signature comparison mode - handled before the Flask setup so it
# runs without webapp.env / NIC_APP_SECRET_KEY / msal. Local matcher only
# (Base64.ai is still called if configured, like the web flow).
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--compare":
    sys.exit(_cli_compare(sys.argv[2:]))

import uuid
from functools import wraps

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

# -----------------------------------------------------------------------------
# Load this web app's own settings (secret key, allowed users, port, etc).
# This is separate from the PRD/LIME credentials loaded up in the
# CONFIGURATION section above.
#
# Resolution order:
#   1. NIC_WEBAPP_ENV env var, if set (absolute path).
#   2. D:\ENV\webapp.env (fixed local-dev location).
# As above, in a container these are usually injected as real environment
# variables and load_dotenv becomes a harmless no-op.
# -----------------------------------------------------------------------------
_WEBAPP_ENV = os.environ.get("NIC_WEBAPP_ENV") or r"D:\ENV\webapp.env"
if os.path.isfile(_WEBAPP_ENV):
    load_dotenv(_WEBAPP_ENV)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif", "webp"}
MAX_UPLOAD_MB = int(os.environ.get("NIC_APP_MAX_UPLOAD_MB", "15"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Comparison-result keys that may carry face_model / face_error, in priority
# order - checked by _attach_face_engine_info() / _attach_lime_identity_fields()
# below when building the JSON response.
_COMPARISON_KEYS = (
    "stored_prd_comparison",
    "capture_prd_comparison",
    "lime_prd_comparison",
    "stored_lime_comparison",
    "capture_lime_comparison",
)

app = Flask(__name__)
app.secret_key = os.environ.get("NIC_APP_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError(
        "NIC_APP_SECRET_KEY is not set. Generate one (e.g. `python -c "
        "\"import secrets;print(secrets.token_hex(32))\"`) and put it in webapp.env."
    )
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# This app is meant to run behind an nginx reverse proxy that terminates
# HTTPS (see DEPLOY.md) - ProxyFix lets Flask see the real scheme/IP from
# the X-Forwarded-* headers nginx sets, and the cookie flags below stop the
# session cookie from ever being sent over plain HTTP once that's in place.
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("NIC_APP_HTTPS", "true").lower() == "true"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"



# -----------------------------------------------------------------------------
# Access control - specific allowed users only
# -----------------------------------------------------------------------------
def _load_allowed_users():
    """
    NIC_APP_USERS format: "username:pbkdf2-hash,username2:pbkdf2-hash2"
    Generate each hash with hash_password.py - passwords are never stored
    or logged in plaintext.
    """
    raw = os.environ.get("NIC_APP_USERS", "")
    users = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        username, pw_hash = pair.split(":", 1)
        username = username.strip()
        if username:
            users[username] = pw_hash.strip()
    return users


ALLOWED_USERS = _load_allowed_users()


# -----------------------------------------------------------------------------
# Microsoft Entra ID (Azure AD) SSO - optional, alongside local login above.
# Local NIC_APP_USERS accounts still work (e.g. for a backup/admin login);
# this block only activates the "Sign in with Microsoft" path when the
# required ENTRA_* env vars are set.
# -----------------------------------------------------------------------------
import msal

ENTRA_CLIENT_ID = os.environ.get("ENTRA_CLIENT_ID")
ENTRA_CLIENT_SECRET = os.environ.get("ENTRA_CLIENT_SECRET")
ENTRA_TENANT_ID = os.environ.get("ENTRA_TENANT_ID")
ENTRA_REDIRECT_PATH = "/auth/callback"
ENTRA_SCOPE = ["User.Read"]

# Defense-in-depth allowlist on top of Entra ID sign-in itself. Leave both
# unset to allow any user who can sign in to your tenant; the recommended
# primary control is still restricting this app in the Entra ID Enterprise
# Application's "Users and groups" (Assignment required = Yes) - see
# DEPLOY.md.
ENTRA_ALLOWED_DOMAIN = os.environ.get("ENTRA_ALLOWED_DOMAIN", "").strip().lower()
ENTRA_ALLOWED_USERS = {
    u.strip().lower() for u in os.environ.get("ENTRA_ALLOWED_USERS", "").split(",") if u.strip()
}

SSO_ENABLED = bool(ENTRA_CLIENT_ID and ENTRA_CLIENT_SECRET and ENTRA_TENANT_ID)
ENTRA_AUTHORITY = f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}" if SSO_ENABLED else ""


def _build_msal_app(cache=None):
    return msal.ConfidentialClientApplication(
        ENTRA_CLIENT_ID,
        authority=ENTRA_AUTHORITY,
        client_credential=ENTRA_CLIENT_SECRET,
        token_cache=cache,
    )


def _sso_user_allowed(email, upn):
    candidates = {v.lower() for v in (email, upn) if v}
    if ENTRA_ALLOWED_USERS and not (candidates & ENTRA_ALLOWED_USERS):
        return False
    if ENTRA_ALLOWED_DOMAIN and not any(c.endswith("@" + ENTRA_ALLOWED_DOMAIN) for c in candidates):
        return False
    return True


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not ALLOWED_USERS:
            error = "No local users are configured for this app yet. See DEPLOY.md."
        else:
            stored_hash = ALLOWED_USERS.get(username)
            if stored_hash and check_password_hash(stored_hash, password):
                session.clear()
                session["user"] = username
                session["auth_method"] = "local"
                next_url = request.args.get("next") or url_for("index")
                return redirect(next_url)
            error = "Invalid username or password."
    return render_template("login.html", error=error, sso_enabled=SSO_ENABLED)


@app.route("/login/microsoft")
def login_microsoft():
    if not SSO_ENABLED:
        return redirect(url_for("login"))
    session["sso_flow_state"] = os.urandom(16).hex()
    session["sso_next"] = request.args.get("next") or url_for("index")
    msal_app = _build_msal_app()
    auth_url = msal_app.get_authorization_request_url(
        ENTRA_SCOPE,
        state=session["sso_flow_state"],
        redirect_uri=url_for("auth_callback", _external=True),
    )
    return redirect(auth_url)


@app.route("/auth/callback")
def auth_callback():
    if not SSO_ENABLED:
        return redirect(url_for("login"))

    if request.args.get("state") != session.get("sso_flow_state"):
        return render_template("login.html", error="Sign-in session expired - please try again.", sso_enabled=SSO_ENABLED)

    if "error" in request.args:
        return render_template(
            "login.html",
            error=f"Microsoft sign-in failed: {request.args.get('error_description', request.args['error'])}",
            sso_enabled=SSO_ENABLED,
        )

    code = request.args.get("code")
    if not code:
        return render_template("login.html", error="No authorization code received from Microsoft.", sso_enabled=SSO_ENABLED)

    msal_app = _build_msal_app()
    result = msal_app.acquire_token_by_authorization_code(
        code,
        scopes=ENTRA_SCOPE,
        redirect_uri=url_for("auth_callback", _external=True),
    )

    if "error" in result:
        return render_template(
            "login.html",
            error=f"Microsoft sign-in failed: {result.get('error_description', result['error'])}",
            sso_enabled=SSO_ENABLED,
        )

    claims = result.get("id_token_claims", {})
    email = (claims.get("email") or "").strip()
    upn = (claims.get("preferred_username") or "").strip()
    display_name = claims.get("name") or email or upn

    if not _sso_user_allowed(email, upn):
        return render_template(
            "login.html",
            error="Your Microsoft account is not authorized for this app. Contact an administrator.",
            sso_enabled=SSO_ENABLED,
        )

    next_url = session.get("sso_next") or url_for("index")
    session.clear()
    session["user"] = display_name
    session["auth_method"] = "sso"
    return redirect(next_url)


@app.route("/logout")
def logout():
    was_sso = session.get("auth_method") == "sso"
    session.clear()
    if was_sso and SSO_ENABLED:
        logout_url = (
            f"{ENTRA_AUTHORITY}/oauth2/v2.0/logout"
            f"?post_logout_redirect_uri={url_for('login', _external=True)}"
        )
        return redirect(logout_url)
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template("index.html", user=session.get("user"))


# -----------------------------------------------------------------------------
# AI engine status - same two subsystems surfaced in the desktop GUI
# -----------------------------------------------------------------------------
@app.route("/api/status")
@login_required
def api_status():
    face = {"available": False, "label": "Unknown"}
    try:
        agent = get_face_agent()
        if agent.is_available():
            face["available"] = True
            face["label"] = f"Ready ({agent.model_name})"
        else:
            face["label"] = "Not available - install face_recognition, deepface, or insightface"
    except Exception as exc:
        face["label"] = f"Error checking agent: {exc}"

    detector = {"available": False, "label": "Unknown"}
    try:
        d = get_live_face_detector()
        if d.backend == "mediapipe":
            detector.update(available=True, label="Ready (MediaPipe)")
        elif d.backend == "yunet":
            detector.update(available=True, label="Ready (OpenCV YuNet)")
        elif d.backend == "haar":
            detector.update(available=True, label="Ready (Haar Cascade - less accurate)")
        else:
            detector["label"] = "Not available - install mediapipe"
    except Exception as exc:
        detector["label"] = f"Error checking detector: {exc}"

    return jsonify({"face_agent": face, "live_detector": detector})


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _allowed_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def _save_upload(file_storage):
    filename = secure_filename(file_storage.filename or "")
    if not filename or not _allowed_image(filename):
        raise ValueError("Please upload a valid image file (jpg, png, bmp, tiff, gif, or webp).")
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(UPLOAD_DIR, unique_name)
    file_storage.save(path)
    return path


def _decode_captured_photo(data_url):
    if not data_url or not isinstance(data_url, str):
        raise ValueError("No captured photo was received.")
    return decode_base64_image(data_url)


def _attach_face_engine_info(result):
    if result.get("face_model") or result.get("face_error"):
        return
    for key in _COMPARISON_KEYS:
        comparison = result.get(key)
        if isinstance(comparison, dict):
            if comparison.get("face_model") and not result.get("face_model"):
                result["face_model"] = comparison.get("face_model")
            if comparison.get("face_error") and not result.get("face_error"):
                result["face_error"] = comparison.get("face_error")


def _find_field(data, keywords, exclude=()):
    """Recursively search a dict/list for the first string value whose key
    contains any of `keywords` (case-insensitive) and none of `exclude`."""
    if isinstance(data, dict):
        for k, v in data.items():
            lk = k.lower()
            if any(kw in lk for kw in keywords) and not any(ex in lk for ex in exclude):
                if isinstance(v, str) and v.strip():
                    return v.strip()
            if isinstance(v, (dict, list)):
                found = _find_field(v, keywords, exclude)
                if found:
                    return found
    elif isinstance(data, list):
        for item in data:
            found = _find_field(item, keywords, exclude)
            if found:
                return found
    return None


def _attach_lime_identity_fields(result):
    """The LIME-vs-PRD comparison (shown in the UI as 'name: different (73%)'
    etc.) proves the engine already holds both sides' raw values, under a
    ...['fields'][<field>]['lime_value' / 'prd_value'] structure. Depending
    on the verification mode, that 'fields' dict lives either at the top
    level (result['data_comparison']) or nested inside one of the
    *_comparison objects (result['stored_lime_comparison']['data_comparison']
    or ['fields'] directly) - so check every location before falling back
    to a generic search."""

    def _from_fields(fields, *field_names):
        if not isinstance(fields, dict):
            return None
        for fname in field_names:
            entry = fields.get(fname)
            if isinstance(entry, dict):
                val = entry.get("lime_value")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None

    # Gather every plausible 'fields' dict: top-level, and nested inside
    # each *_comparison object (either directly, or under its own
    # 'data_comparison' key).
    field_dicts = [(result.get("data_comparison") or {}).get("fields")]
    for key in _COMPARISON_KEYS:
        comparison = result.get(key)
        if isinstance(comparison, dict):
            field_dicts.append(comparison.get("fields"))
            field_dicts.append((comparison.get("data_comparison") or {}).get("fields"))

    for fields in field_dicts:
        if not result.get("lime_name"):
            name = _from_fields(fields, "name", "full_name", "customer_name")
            if name:
                result["lime_name"] = name
        if not result.get("lime_address"):
            address = _from_fields(fields, "address", "permanent_address", "residential_address", "customer_address")
            if address:
                result["lime_address"] = address

    # Fallback: generic recursive search of the raw comparison objects
    for key in _COMPARISON_KEYS:
        comparison = result.get(key)
        if not isinstance(comparison, dict):
            continue
        if not result.get("lime_name"):
            name = _find_field(comparison, keywords=["name"], exclude=["model", "filename", "doc_type", "document_type"])
            if name:
                result["lime_name"] = name
        if not result.get("lime_address"):
            address = _find_field(comparison, keywords=["address", "addr", "resident", "location"], exclude=["face"])
            if address:
                result["lime_address"] = address

    if result.get("lime_found") and (not result.get("lime_name") or not result.get("lime_address")):
        # Diagnostic only - dump the raw comparison structure to the console
        # so the actual key names holding LIME's name/address (if any) can be found.
        try:
            dump = {k: result.get(k) for k in _COMPARISON_KEYS if isinstance(result.get(k), dict)}
            dump["data_comparison"] = result.get("data_comparison")
            logging.info(
                "LIME name=%r address=%r not fully found. Raw comparison data: %s",
                result.get("lime_name"), result.get("lime_address"),
                json.dumps(dump, default=str, indent=2),
            )
        except Exception:
            pass


def _normalise_prd_result(raw, source_label):
    if not raw.get("success"):
        return raw
    comparison = raw.get("comparison", {})
    return {
        "success": True,
        "final_status": raw.get("status", "ERROR"),
        "final_score": raw.get("score", 0),
        "final_message": comparison.get("message", ""),
        "prd_data": raw.get("prd_data", {}),
        "face_match": raw.get("face_match", False),
        "face_similarity": raw.get("face_similarity", 0),
        "face_model": comparison.get("face_model"),
        "face_error": comparison.get("face_error"),
        "image_source": source_label,
        "lime_found": False,
    }


def _strip_heavy_fields(result):
    """Drop large/base64 image payloads before sending JSON back to the
    browser - the client never needs the raw PRD/LIME photo bytes, and
    sending them back would bloat every response for no benefit."""
    prd = result.get("prd_data")
    if isinstance(prd, dict):
        prd = dict(prd)
        prd.pop("nic_front_image", None)
        prd.pop("nic_back_image", None)
        result = dict(result)
        result["prd_data"] = prd
    for key in _COMPARISON_KEYS:
        if key in result:
            result = dict(result)
            result.pop(key, None)
    result.pop("data_comparison_raw", None)
    return result


# -----------------------------------------------------------------------------
# Main verification endpoint
# -----------------------------------------------------------------------------
@app.route("/api/verify", methods=["POST"])
@login_required
def api_verify():
    mode = request.form.get("mode", "")
    nic = (request.form.get("nic") or "").strip()
    mobile = (request.form.get("mobile") or "").strip()
    tmp_path = None
    username = session.get("user", "-")

    def _log_and_respond(result, http_status=200):
        # Lightweight text log for EVERY search attempt - success, business
        # rejection, validation error, or exception - so nothing is missed.
        # Best-effort: logging is never allowed to block the response.
        try:
            save_verification_search_log(
                username=username,
                nic=nic,
                mobile=mobile,
                result_status=(
                    result.get("final_status")
                    or result.get("status")
                    or ("ERROR" if not result.get("success") else "UNKNOWN")
                ),
            )
        except Exception:
            pass
        return jsonify(result), http_status

    try:
        if mode == "stored_prd":
            image_file = request.files.get("image")
            if not image_file or not nic:
                return _log_and_respond({"success": False, "error": "NIC number and a NIC image are required."}, 400)
            tmp_path = _save_upload(image_file)
            raw = verify_stored_image_with_prd(tmp_path, nic)
            result = _normalise_prd_result(raw, image_file.filename)

        elif mode == "stored_lime":
            image_file = request.files.get("image")
            if not image_file or not nic:
                return _log_and_respond({"success": False, "error": "NIC number and a NIC image are required."}, 400)
            tmp_path = _save_upload(image_file)
            result = verify_stored_image_with_lime_and_prd(tmp_path, nic, mobile or None)

        elif mode == "lime_repository":
            if not nic:
                return _log_and_respond({"success": False, "error": "NIC number is required."}, 400)
            result = verify_lime_repository_with_prd(nic, mobile or None)

        elif mode == "camera_capture":
            data_url = request.form.get("image_data")
            if not data_url or not nic:
                return _log_and_respond({"success": False, "error": "A captured photo and NIC number are required."}, 400)
            image = _decode_captured_photo(data_url)
            if mobile:
                result = verify_with_lime_and_prd(image, mobile, nic)
            else:
                raw = verify_captured_with_prd(image, nic)
                result = _normalise_prd_result(raw, "Browser camera capture")

        else:
            return _log_and_respond({"success": False, "error": "Unknown verification mode."}, 400)

        if result.get("success"):
            _attach_face_engine_info(result)
            _attach_lime_identity_fields(result)
            result = _strip_heavy_fields(result)

        return _log_and_respond(result)

    except Exception as exc:
        return _log_and_respond({"success": False, "error": str(exc)}, 500)

    finally:
        # Uploaded photos are only needed for the duration of the comparison.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# -----------------------------------------------------------------------------
# Signature verification endpoint - same LIME-repository pattern as
# /api/verify's "lime_repository" and "stored_lime" modes above, applied to
# the customer's stored SIGNATURE document instead of KYC_Front/NIC_Front.
# Accepts either an uploaded signature file ("mode=upload", multipart field
# "signature") or a browser-captured signature ("mode=capture", form field
# "signature_data" as a data: URL) - same two input paths the NIC flows use.
# -----------------------------------------------------------------------------
@app.route("/api/verify-signature", methods=["POST"])
@login_required
def api_verify_signature():
    mode = request.form.get("mode", "")
    nic = (request.form.get("nic") or "").strip()
    mobile = (request.form.get("mobile") or "").strip()
    tmp_path = None
    username = session.get("user", "-")

    def _log_and_respond(result, http_status=200):
        try:
            save_verification_search_log(
                username=username,
                nic=nic,
                mobile=mobile,
                result_status=(
                    result.get("final_status")
                    or result.get("status")
                    or ("ERROR" if not result.get("success") else "UNKNOWN")
                ),
            )
        except Exception:
            pass
        return jsonify(result), http_status

    try:
        if not nic:
            return _log_and_respond({"success": False, "error": "NIC number is required."}, 400)

        if mode == "upload":
            image_file = request.files.get("signature")
            if not image_file:
                return _log_and_respond({"success": False, "error": "A signature image file is required."}, 400)
            tmp_path = _save_upload(image_file)
            signature_image = load_local_image(tmp_path)

        elif mode == "capture":
            data_url = request.form.get("signature_data")
            if not data_url:
                return _log_and_respond({"success": False, "error": "A captured signature is required."}, 400)
            signature_image = _decode_captured_photo(data_url)

        else:
            return _log_and_respond({"success": False, "error": "Unknown signature verification mode."}, 400)

        result = verify_signature_with_lime(nic, signature_image, mobile or None)

        return _log_and_respond(result)

    except Exception as exc:
        return _log_and_respond({"success": False, "error": str(exc)}, 500)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    host = os.environ.get("NIC_APP_HOST", "0.0.0.0")
    port = int(os.environ.get("NIC_APP_PORT", "8085"))
    # Debug must be OFF unless explicitly enabled. (The previous
    # "== 'false'" comparison meant debug defaulted to True, which is
    # unsafe for anything but local experimentation.)
    debug = os.environ.get("NIC_APP_DEBUG", "false").lower() == "true"

    # Self-signed HTTPS support (LOCAL DEV ONLY - see DEPLOY.md for prod,
    # which terminates TLS at nginx instead). Paths are overridable via env
    # vars so a real deployment can mount a cert/key from outside the app
    # folder (e.g. a Kubernetes Secret at /etc/nic-app/tls/) instead of
    # relying on files that live in BASE_DIR.
    #
    # SECURITY: never commit cert.pem/key.pem or let them into the Docker
    # build context - anything COPY'd from this folder ends up baked into
    # the image layer, including private keys. Keep them out of BASE_DIR
    # in any environment where this code gets containerized, and make sure
    # .dockerignore excludes *.pem (see .dockerignore).
    cert_file = os.environ.get("NIC_APP_TLS_CERT", r"D:\ENV\cert.pem")
    key_file = os.environ.get("NIC_APP_TLS_KEY", r"D:\ENV\key.pem")
    ssl_context = None
    if os.path.isfile(cert_file) and os.path.isfile(key_file):
        ssl_context = (cert_file, key_file)
    else:
        print(
            "WARNING: TLS cert/key not found (NIC_APP_TLS_CERT/NIC_APP_TLS_KEY, "
            "or cert.pem/key.pem next to app.py) - running WITHOUT HTTPS. "
            "Run generate_cert.py for local dev, or mount a real cert/key in prod."
        )

    app.run(host=host, port=port, debug=debug, ssl_context=ssl_context)



#from flask import Flask
#
#app = Flask(__name__)
#
#@app.route("/")
#def home():
#    return "Flask App Running on Azure ..."
#
#if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=8000)