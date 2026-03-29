import base64
import json
import math
import mimetypes
import os
import random
import shutil
import sqlite3
from datetime import datetime

import requests
from PIL import Image
from flask import Flask, request, jsonify, render_template, url_for, flash, redirect, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from FaceRecognizationMode import face_train_model, predict_face, compare_faces
from Palm_Recognization import palm_predict_image, palm_train_model, palm_similarity
from Voice_Recognization import Extract_Text

app = Flask(__name__)
app.secret_key = "supersecretkey"
global voicefilename
global save_path
global hospitals
global ambulances
# Base folders
BASE_DIR = "Biometric_Data"
FACE_DIR = os.path.join(BASE_DIR, "faces")
PALM_DIR = os.path.join(BASE_DIR, "palms")
VOICE_DIR = os.path.join(BASE_DIR, "voices")

# Create base folders
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(PALM_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)
DATABASE = "database.db"


def get_db():
    return sqlite3.connect(DATABASE)


# Initialize database
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            user_id INTEGER UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            voicemessage TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


# Initialize database
def createemergency_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
           CREATE TABLE IF NOT EXISTS emergency_appointments (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               patient_name TEXT NOT NULL,
               mobile TEXT NOT NULL,
               lat REAL NOT NULL,
               lng REAL NOT NULL,
               heart_rate INTEGER,
               spo2 INTEGER,
               ambulance TEXT,
               hospital TEXT,
               severity TEXT,
               status TEXT DEFAULT 'Pending',
               email_sent INTEGER DEFAULT 0,
               createdatetime TEXT DEFAULT CURRENT_TIMESTAMP
           )
       """)
    conn.commit()
    conn.close()


createemergency_db()


# Create admitted_patients table (if not exists)
def init_appointmentdb():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS admitted_patients (id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER,patient_name TEXT NOT NULL,age INTEGER NOT NULL,gender TEXT NOT NULL, blood_group TEXT NOT NULL,phone TEXT NOT NULL,emergency_contact TEXT NOT NULL,email TEXT NOT NULL, address TEXT NOT NULL,symptoms TEXT NOT NULL,diagnosis TEXT NOT NULL,admission_reason TEXT NOT NULL,doctor_assigned TEXT NOT NULL,admission_date TEXT NOT NULL,scheduled_datetime TEXT,status TEXT DEFAULT 'Pending',created_at TEXT NOT NULL)")

    conn.commit()
    conn.close()


init_appointmentdb()


# GET DOCTORS BASED ON HOSPITAL
@app.route("/get_doctors", methods=["POST"])
def get_doctors():
    hospital = request.form.get("hospital")

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        "SELECT doctor_name, specialization FROM doctors WHERE hospital_name=?",
        (hospital,)
    )

    doctors = cur.fetchall()

    conn.close()

    doctor_list = []

    for d in doctors:
        doctor_list.append({
            "name": d["doctor_name"],
            "specialization": d["specialization"]
        })

    return jsonify(doctor_list)


# Endpoint to receive coordinates
@app.route('/save_location', methods=['POST'])
def save_location():
    global final_lat, final_lng
    data = request.json
    final_lat = data.get('lat')
    final_lng = data.get('lng')

    # Save to a file (or database)
    with open('patient_location.txt', 'w') as f:
        f.write(f"{final_lat},{final_lng}\n")
    return jsonify({"status": "success"}), 200


# ------------------------------
# 🚑 Ambulance Initial Locations
# ------------------------------

with open("patient_location.txt", "r") as file:
    line = file.readline().strip()  # "18.78941108504643,78.29367092275112"
    lat_str, lng_str = line.split(",")
    final_lat = float(lat_str)
    final_lng = float(lng_str)


# ------------------------------
# Home Page
# ------------------------------

@app.route("/ambulance_map")
def ambulance_map():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT hospital_name FROM doctors")
    hospitals = cur.fetchall()

    print(hospitals)  # DEBUG

    conn.close()
    return render_template("doctor_slots.html", hospitals=hospitals)


@app.route("/emergencydashboard")
def emergencydashboard():
    return render_template("emergency_dashboard.html")


hospitals = {}
ambulances = {}
doctors = {}


# ------------------------------------------------
# Overpass Request Helper
# ------------------------------------------------

def overpassRequest(query):
    servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter"
    ]

    for s in servers:
        try:

            r = requests.get(s, params={'data': query}, timeout=25)

            if r.status_code == 200:
                return r.json()

        except:
            continue

    print("All Overpass servers failed")
    return {"elements": []}


# ------------------------------------------------
# Find Hospitals
# ------------------------------------------------

def findHospitals(lat, lng):
    radius = 5000

    query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:{radius},{lat},{lng});
      way["amenity"="hospital"](around:{radius},{lat},{lng});
    );
    out center;
    """

    data = overpassRequest(query)

    result = {}
    count = 1

    for el in data["elements"]:

        name = el.get("tags", {}).get("name", "Hospital_" + str(count))

        if "lat" in el:
            lat1 = el["lat"]
            lng1 = el["lon"]
        else:
            lat1 = el["center"]["lat"]
            lng1 = el["center"]["lon"]

        result[name] = {
            "lat": lat1,
            "lng": lng1
        }

        count += 1

    return result


# ------------------------------------------------
# Find Ambulances
# ------------------------------------------------

def findAmbulances(lat, lng):
    radius = 5000

    query = f"""
    [out:json];
    (
      node["amenity"="ambulance_station"](around:{radius},{lat},{lng});
      node["emergency"="ambulance_station"](around:{radius},{lat},{lng});
    );
    out;
    """

    data = overpassRequest(query)

    result = {}
    count = 1

    for el in data["elements"]:
        name = el.get("tags", {}).get("name", "Ambulance_" + str(count))

        result[name] = {
            "lat": el["lat"],
            "lng": el["lon"],
            "status": "Available"
        }

        count += 1

    return result


# ------------------------------------------------
# Find Doctors
# ------------------------------------------------

def findDoctors(lat, lng):
    radius = 5000

    query = f"""
    [out:json];
    (
      node["amenity"="doctors"](around:{radius},{lat},{lng});
      node["healthcare"="doctor"](around:{radius},{lat},{lng});
      node["amenity"="clinic"](around:{radius},{lat},{lng});
    );
    out;
    """

    data = overpassRequest(query)

    result = {}
    count = 1

    for el in data["elements"]:
        tags = el.get("tags", {})

        name = tags.get("name", "Doctor_" + str(count))
        specialization = tags.get("healthcare:speciality", "General")

        result[name] = {

            "lat": el["lat"],
            "lng": el["lon"],
            "specialization": specialization,
            "hospital": tags.get("operator", "Unknown Hospital")

        }

        count += 1

    return result


# ------------------------------------------------
# Distance Function
# ------------------------------------------------

def distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


# ------------------------------------------------
# Initialize System
# ------------------------------------------------

@app.route("/init_system", methods=["POST"])
def init_system():
    data = request.json

    lat = data["lat"]
    lng = data["lng"]

    global hospitals
    global ambulances
    global doctors

    hospitals = findHospitals(lat, lng)
    ambulances = findAmbulances(lat, lng)
    doctors = findDoctors(lat, lng)

    return jsonify({
        "hospitals": hospitals,
        "ambulances": ambulances,
        "doctors": doctors
    })


# ------------------------------------------------
# Simulate Ambulance Movement
# ------------------------------------------------

@app.route("/get_ambulances")
def get_ambulances():
    for a in ambulances:
        ambulances[a]["lat"] += random.uniform(-0.0005, 0.0005)
        ambulances[a]["lng"] += random.uniform(-0.0005, 0.0005)

    return jsonify(ambulances)


# ------------------------------------------------
# AI Severity Prediction
# ------------------------------------------------

def predictSeverity(heart, spo2):
    if spo2 < 85 or heart > 140:
        return "Critical"

    elif spo2 < 92 or heart > 120:
        return "Serious"

    else:
        return "Stable"


# ------------------------------------------------
# Find Nearest Ambulance
# ------------------------------------------------

def nearestAmbulance(lat, lng):
    best = None
    bestDist = 999

    for a in ambulances:

        if ambulances[a]["status"] == "Available":

            d = distance(
                lat, lng,
                ambulances[a]["lat"],
                ambulances[a]["lng"]
            )

            if d < bestDist:
                bestDist = d
                best = a

    return best


# ------------------------------------------------
# Find Nearest Hospital
# ------------------------------------------------

def nearestHospital(lat, lng):
    best = None
    bestDist = 999

    for h in hospitals:

        d = distance(
            lat, lng,
            hospitals[h]["lat"],
            hospitals[h]["lng"]
        )

        if d < bestDist:
            bestDist = d
            best = h

    return best


# ------------------------------------------------
# Emergency Report
# ------------------------------------------------

@app.route("/report_emergency", methods=["POST"])
def report_emergency():
    data = request.json
    patient_name = data["patient_name"]
    mobile = data["mobile"]
    lat = data["lat"]
    lng = data["lng"]

    heart = data["heart_rate"]
    spo2 = data["spo2"]

    severity = predictSeverity(heart, spo2)

    amb = nearestAmbulance(lat, lng)

    hosp = nearestHospital(lat, lng)

    if amb:
        ambulances[amb]["status"] = "Dispatched"
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
            INSERT INTO emergency_appointments
            (patient_name, mobile, lat, lng, heart_rate, spo2, ambulance, hospital, severity, createdatetime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
        patient_name, mobile, lat, lng, heart, spo2,
        amb, hosp, severity,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

    return jsonify({

        "severity": severity,
        "ambulance": amb,
        "hospital": hosp

    })


# ---------------- Routes ---------------- #

@app.route('/')
@app.route('/Home')
def homepage():
    return render_template('index.html')


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/appointments')
def appointments():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT hospital_name FROM doctors")
    hospitals = cur.fetchall()

    conn.close()
    return render_template("appointments.html", hospitals=hospitals)


@app.route('/medical')
def medical():
    return render_template('medical.html')


@app.route('/nutrition')
def nutrition():
    return render_template('nutrition.html')


@app.route('/scan')
def scan():
    return render_template('scan.html')


@app.route('/logout')
def logout():
    # Example: redirect to login page after logout
    return redirect(url_for('dashboard'))


@app.route('/Registration')
def registration():
    return render_template('signup.html')


@app.route('/Login')
def login_page():
    return render_template('signin.html')


@app.route('/userhomepage')
def userhomepage():
    return render_template('Userhome.html')


@app.route('/admin_login1')
def admin_login1():
    return render_template('admin_login.html')


# ---------------- Helper Functions ---------------- #

def save_image(username, biometric_type, image_data):
    """Save Base64 images with rotation (0 → 5)."""

    if biometric_type not in ["face", "palm"]:
        return None, "Invalid biometric type"

    dir_map = {"face": FACE_DIR, "palm": PALM_DIR}
    user_folder = os.path.join(dir_map[biometric_type], username)
    os.makedirs(user_folder, exist_ok=True)

    try:
        header, encoded = image_data.split(",", 1)
        binary_data = base64.b64decode(encoded)

        existing_files = sorted([
            f for f in os.listdir(user_folder)
            if f.startswith(username) and f.endswith(".jpg")
        ])

        if len(existing_files) < 6:
            index = len(existing_files)
        else:
            # ✅ Rotate → remove oldest
            oldest_file = existing_files[0]
            os.remove(os.path.join(user_folder, oldest_file))
            index = 5  # keep max index consistent

        filepath = os.path.join(user_folder, f"{username}_{index}.jpg")

        with open(filepath, "wb") as f:
            f.write(binary_data)

        print(f"✅ {biometric_type.upper()} saved: {filepath}")
        return filepath, None

    except Exception as e:
        return None, str(e)


def save_voice(username, audio_file):
    """Save voice file to user's folder."""
    user_folder = os.path.join(VOICE_DIR, username)
    os.makedirs(user_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filepath = os.path.join(user_folder, f"voice_{timestamp}.wav")
    try:
        audio_file.save(filepath)
        print(f"✅ VOICE saved: {filepath}")
        return filepath, None
    except Exception as e:
        return None, str(e)


def trigger_model_update(username, biometric_type, filepath):
    global voicefilename
    voicefilename = filepath
    """Placeholder: Trigger model training / update after capture."""
    print(f"📌 Triggering model update for {username} ({biometric_type}) -> {filepath}")
    # Here you can call your KNN/CNN/voice training pipeline


# ---------------- Routes for Biometric Capture ---------------- #

@app.route('/biometric_capture', methods=['POST'])
def biometric_capture():
    data = request.json
    image_data = data.get("image")
    biometric_type = data.get("type")
    username = data.get("username")

    if not image_data or not username:
        return jsonify({"status": "error", "message": "Missing username or image"}), 400

    filepath, error = save_image(username, biometric_type, image_data)
    if error:
        return jsonify({"status": "error", "message": error}), 500

    trigger_model_update(username, biometric_type, filepath)
    return jsonify({"status": "success", "path": filepath})


@app.route('/voice_capture', methods=['POST'])
def voice_capture():
    audio_file = request.files.get('voice')
    username = request.form.get("username")

    if not audio_file or not username:
        return jsonify({"status": "error", "message": "Missing username or audio"}), 400

    filepath, error = save_voice(username, audio_file)
    if error:
        return jsonify({"status": "error", "message": error}), 500

    trigger_model_update(username, "voice", filepath)
    return jsonify({"status": "success", "path": filepath})


# ---------------- User Registration ---------------- #

@app.route('/add', methods=['POST'])
def add_user():
    username = request.form.get("newusername")
    user_id = request.form.get("newuserid")
    password = request.form.get("newuserpassword")
    email = request.form.get("newuseremail")
    global voicefilename
    if not all([username, user_id, password, email]):
        flash("All fields are required.", "danger")
        return redirect(url_for("registration"))

    try:
        with sqlite3.connect("database.db") as conn:
            # Create user folders for biometrics
            os.makedirs(os.path.join(FACE_DIR, username), exist_ok=True)
            os.makedirs(os.path.join(PALM_DIR, username), exist_ok=True)
            os.makedirs(os.path.join(VOICE_DIR, username), exist_ok=True)
            voicemessage = Extract_Text(voicefilename)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, user_id, password, email,voicemessage)
                VALUES (?, ?, ?, ?, ?)
            """, (username, user_id, password, email, voicemessage))
            conn.commit()

        face_train_model()
        palm_train_model()
        flash("User registered successfully!", "success")
        return redirect(url_for("registration"))

    except sqlite3.IntegrityError as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for("registration"))


SAVE_FOLDER = "Input_Data"
os.makedirs(SAVE_FOLDER, exist_ok=True)


# ✅ Save Face & Palm Images
@app.route('/login_biometric_capture', methods=['POST'])
def login_biometric_capture():
    data = request.get_json()

    image_data = data.get("image")
    capture_type = data.get("type")

    if not all([image_data, capture_type]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    try:
        # Remove base64 header
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        filename = f"{capture_type}_1.jpg"
        save_path = os.path.join(SAVE_FOLDER, filename)

        with open(save_path, "wb") as f:
            f.write(image_bytes)

        print(f"{capture_type.upper()} saved:", save_path)

        return jsonify({"status": "success"})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"status": "error"}), 500


# ✅ Save Voice File
@app.route('/login_voice_capture', methods=['POST'])
def login_voice_capture():
    global save_path
    voice_file = request.files.get("voice")

    if not voice_file:
        return jsonify({"status": "error", "message": "No voice"}), 400

    try:
        save_path = os.path.join(SAVE_FOLDER, "voice_1.wav")

        voice_file.save(save_path)

        print("VOICE saved:", save_path)

        return jsonify({"status": "success"})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"status": "error"}), 500


# ---------------- Optional: User Login ---------------- #


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get("newusername")
    password = request.form.get("userpassword")
    authtype = request.form.get("authtype")
    global save_path
    if not username or not password:
        flash("Enter username and password.", "danger")
        return redirect(url_for("login_page"))
    voice_message = None
    with sqlite3.connect("database.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password,voicemessage FROM users WHERE username=? and password=?", (username, password))
        result = cursor.fetchone()
        voice_message = result[1]
        if result is None:
            flash("Invalid username or password", "danger")
            return redirect(url_for("login_page"))
    if authtype == "face":
        import os
        imagepath = "Input_Data\\face_1.jpg"
        imagepath_2 = os.path.join(
            "Biometric_Data",
            "faces",
            username,
            username + "_1.jpg"
        )
        matched, confidence_sc = compare_faces(imagepath_2, imagepath)
        face_predicted_user, confidence = predict_face(imagepath)
        print(f"Predicted User: {face_predicted_user}")
        print(confidence)
        if (face_predicted_user == username and confidence >= 95.0) or matched == True:
            return render_template("UserHome.html",
                                   username=username,
                                   confidence=confidence_sc)
        else:
            flash("Face is Mismatched!!Try Again!!", "danger")
            return redirect(url_for("login_page"))
    elif authtype == "palm":
        imagepath = "Input_Data/palm_1.jpg"
        import os
        imagepath_2 = os.path.join(
            "Biometric_Data",
            "palms",
            username,
            username + "_1.jpg"
        )
        similarity_sc, confidence_score = palm_similarity(imagepath_2, imagepath)
        print(imagepath)
        palm_predicted_user, confidence = palm_predict_image(imagepath)
        print(f"Predicted User: {palm_predicted_user}")
        if (palm_predicted_user == username and confidence >= 0.95) or confidence_score >= 75.0:
            return render_template("UserHome.html",
                                   username=username,
                                   confidence=confidence_score)
        else:
            flash("Palm is Mismatched!!Try Again!!", "danger")
            return redirect(url_for("login_page"))
    else:
        print("voice")
        print(save_path)
        voice_message1 = voice_message
        voice_message2 = Extract_Text(save_path)
        # Convert texts to TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([voice_message1, voice_message2])

        # Compute cosine similarity
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        print(f"Similarity score: {similarity_score:.2f}")
        if similarity_score >= 0.8:
            return render_template("UserHome.html",
                                   username=username,
                                   face_image=save_path,
                                   confidence=similarity_score)
        else:
            flash("Voice is Mismatched!!Try Again!!", "danger")
            return redirect(url_for("login_page"))


import os
import numpy as np
import cv2
import google.generativeai as genai
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

CORS(app)

UPLOAD_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔑 Gemini API Key
genai.configure(api_key="AIzaSyBvhGwRI94z8g-KBignyo1kn7GZIMJYFRk")
chat_model = genai.GenerativeModel("gemini-flash-latest")


def predict_disease_from_mri(image_path):
    try:
        # Open MRI image
        image = Image.open(image_path)

        # Prompt for medical analysis
        prompt = """
        You are a medical AI assistant.
        Analyze this MRI scan image carefully.
        Identify possible abnormalities or diseases.
        Provide:
        1. Possible Disease Name
        2. Confidence Level (Low/Medium/High)
        3. Explanation
        4. Recommended Next Steps

        Important: This is not a final medical diagnosis.
        """

        # Send image + prompt to Gemini
        response = chat_model.generate_content([prompt, image])

        return response.text

    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/predict_mri", methods=["POST"])
def predict_mri():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        result = predict_disease_from_mri(filepath)
        print(result)
        return render_template("scan.html",
                               result_r=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🤖 Medical Chatbot (Gemini)
# ==============================

@app.route("/query", methods=["POST"])
def chat():
    user_message = request.json["message"]

    response = chat_model.generate_content(user_message)

    return jsonify({
        "reply": response.text
    })


# ==============================
# 🚨 Emergency AI Assistance
# ==============================

@app.route("/emergency_chat", methods=["POST"])
def emergency_chat():
    disease = request.json["disease"]

    prompt = f"""
    Patient diagnosed with {disease}.
    Provide immediate medical first-aid advice and emergency instructions.
    """

    response = chat_model.generate_content(prompt)

    return jsonify({
        "emergency_advice": response.text
    })


@app.route("/analyze_food", methods=["POST"])
def analyze_food():
    nutrition_result = None
    image_file = ""
    if request.method == "POST" and request.files.get("file"):
        image_file = request.files["file"]
    print(image_file.filename)
    try:
        prompt = """
     You are a nutrition expert. 
     Analyze the uploaded food image and return:

     - Ingredients detected
     - Calories (approximate)
     - Nutritional Breakdown (Carbs, Protein, Fat, Fiber, Sugar)
     - Health Suggestion (Healthy / Unhealthy and why)
     """

        # Convert Django InMemoryUploadedFile -> Gemini-compatible format

        mime_type, _ = mimetypes.guess_type(image_file.name)
        image_data = {
            "mime_type": mime_type or "image/png",
            "data": image_file.read()}
        response = chat_model.generate_content([prompt, image_data])
        nutrition_result = response.text.strip()
        print("Dd")
        print(nutrition_result)

        return render_template("nutrition.html",
                               nutrition_r=nutrition_result)
    except Exception as e:
        print(e)
    return render_template("nutrition.html",
                           nutrition_r=nutrition_result)


from flask import session


@app.route("/admission", methods=["GET", "POST"])
def admission_form():
    if request.method == "POST":
        form_data = [
            request.form.get("patientid"),
            request.form.get("patient_name"),
            request.form.get("age"),
            request.form.get("gender"),
            request.form.get("blood_group"),
            request.form.get("phone"),
            request.form.get("hospital"),
            request.form.get("email"),
            request.form.get("address"),
            request.form.get("symptoms"),
            request.form.get("diagnosis"),
            request.form.get("admission_reason"),
            request.form.get("doctor_assigned"),
            request.form.get("admission_date")
        ]
        print(form_data)
        if any(not field for field in form_data):
            flash("⚠️ All fields are mandatory!")
            return redirect(url_for("appointments"))

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO admitted_patients (
                user_id,patient_name, age, gender, blood_group, phone,
                emergency_contact,email,address, symptoms,
                diagnosis, admission_reason, doctor_assigned,
                admission_date, created_at
            )
            VALUES (?,?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
        """, (*form_data, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        conn.commit()
        conn.close()

        return redirect(url_for("success"))

    return render_template("appointments.html")


@app.route("/success")
def success():
    return render_template("success.html")


# -----------------------------
# ADMIN LOGIN
# -----------------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin":
            session["admin"] = True
            print("ddd")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid Credentials!")

    return render_template("admin_login.html")


# -----------------------------
# ADMIN DASHBOARD (Search + Filter)
# -----------------------------
@app.route("/admin_dashboard")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    search = request.args.get("search", "")
    status_filter = request.args.get("status", "")

    conn = get_db()
    cursor = conn.cursor()

    query = "SELECT * FROM admitted_patients WHERE 1=1"
    params = []

    if search:
        query += " AND patient_name LIKE ?"
        params.append(f"%{search}%")

    if status_filter:
        query += " AND status=?"
        params.append(status_filter)

    query += " ORDER BY created_at DESC"

    cursor.execute(query, params)
    patients = cursor.fetchall()
    conn.close()

    return render_template("admin_dashboard.html", patients=patients)


# -----------------------------
# APPROVE WITH SCHEDULE + EMAIL
# -----------------------------
@app.route("/approve/<int:id>", methods=["POST"])
def approve(id):
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    scheduled_datetime = request.form.get("scheduled_datetime")

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Get doctor
    cursor.execute("SELECT doctor_assigned FROM admitted_patients WHERE id=?", (id,))
    doctor = cursor.fetchone()[0]

    # Prevent double booking
    cursor.execute("""
        SELECT * FROM admitted_patients
        WHERE doctor_assigned=? AND scheduled_datetime=? AND status='Approved'
    """, (doctor, scheduled_datetime))

    if cursor.fetchone():
        conn.close()
        return "❌ Slot already reserved!"

    cursor.execute("""
        UPDATE admitted_patients
        SET status='Approved', scheduled_datetime=?
        WHERE id=?
    """, (scheduled_datetime, id))

    cursor.execute("SELECT email, patient_name FROM admitted_patients WHERE id=?", (id,))
    patient = cursor.fetchone()

    conn.commit()
    conn.close()

    if patient:
        send_email(patient[0], patient[1], scheduled_datetime)

    return redirect(url_for("admin_dashboard"))


# -----------------------------
# REJECT
# -----------------------------
@app.route("/reject/<int:id>")
def reject(id):
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("UPDATE adm SET status='Rejected' WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect(url_for("admin_dashboard"))


# -----------------------------
# LOGOUT
# -----------------------------
@app.route("/adminlogout")
def adminlogout():
    session.pop("admin", None)
    return redirect(url_for("admin_login"))


# -----------------------------
# EMAIL FUNCTION
# -----------------------------
import smtplib
from email.mime.text import MIMEText


def send_email(to_email, patient_name, scheduled_datetime):
    try:
        import smtplib
        from email.mime.text import MIMEText
        subject = "Appointment Approved"
        sender_email = "vadde.seetha@gmail.com"
        sender_password = "qoif fbmc fxst nrhp"
        recipients = to_email

        body = f"""
    Dear {patient_name},

    Your appointment has been APPROVED.

    Scheduled Date & Time:
    {scheduled_datetime}

    Please arrive 15 minutes early.

    Regards,
    Hospital Admin
    """
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(sender_email, sender_password)
            smtp_server.sendmail(sender_email, recipients, msg.as_string())
        print("Message sent!")
    except Exception as e:
        print("Email failed:", e)


# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ===============================
# ANALYTICS
# ===============================


@app.route("/admin/analytics")
def analytics():
    if "admin" not in session:
        return redirect("/admin")

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT status, COUNT(*) FROM admitted_patients GROUP BY status")
    result = cursor.fetchall()

    # Default counts
    counts = {
        "Pending": 0,
        "Approved": 0,
        "Rejected": 0
    }

    for status, count in result:
        counts[status] = count
    print(counts["Approved"])
    return render_template(
        "analytics.html",
        pending=counts["Pending"],
        approved=counts["Approved"],
        rejected=counts["Rejected"]
    )
    conn.close()


# ===============================
# PDF RECEIPT
# ===============================
@app.route("/receipt/<int:id>")
def receipt(id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    c.execute("SELECT * FROM admitted_patients WHERE id=?", (id,))
    p = c.fetchone()
    conn.close()

    filename = f"receipt_{id}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Appointment Receipt</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    fields = [
        f"Patient ID: {p[1]}",
        f"Patient Name: {p[2]}",
        f"Age: {p[3]}",
        f"Doctor: {p[13]}",
        f"Hospital Name: {p[7]}",
        f"Admission Date: {p[14]}",
        f"Scheduled Date: {p[15]}",
        f"Status: {p[16]}"
    ]

    for field in fields:
        elements.append(Paragraph(field, styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

    doc.build(elements)

    return send_file(filename, as_attachment=True)


@app.route("/track", methods=["GET", "POST"])
def track():
    appointment = None

    if request.method == "POST":
        email = request.form["email"]

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM admitted_patients WHERE user_id=? ORDER BY id DESC LIMIT 1", (email,))
        appointment = cursor.fetchone()

        conn.close()

    return render_template("track.html", appointment=appointment)


from datetime import datetime, timedelta


def generate_slots(date):
    slots = []
    start = datetime.strptime(date + " 09:00", "%Y-%m-%d %H:%M")
    end = datetime.strptime(date + " 17:00", "%Y-%m-%d %H:%M")

    while start < end:
        slots.append(start.strftime("%Y-%m-%d %H:%M"))
        start += timedelta(minutes=30)

    return slots


@app.route("/delete/<int:id>")
def delete(id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM admitted_patients WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect("/admin_dashboard")


# =============================
# LOAD AVAILABLE SLOTS
# =============================


from datetime import datetime, timedelta


@app.route("/slots/<int:id>")
def slots(id):
    if "admin" not in session:
        return redirect("/admin_login")

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Get doctor and admission date
    cursor.execute(
        "SELECT doctor_assigned, admission_date FROM admitted_patients WHERE id=?",
        (id,)
    )

    row = cursor.fetchone()

    if not row:
        conn.close()
        return "Patient not found"

    doctor = row[0]
    admission_date = row[1]
    dt = datetime.strptime(admission_date, "%Y-%m-%dT%H:%M")
    admission_date = dt.strftime("%Y-%m-%d")
    # Generate slots
    start = datetime.strptime(admission_date + " 09:00", "%Y-%m-%d %H:%M")
    end = datetime.strptime(admission_date + " 17:00", "%Y-%m-%d %H:%M")

    all_slots = []

    while start <= end:
        all_slots.append(start.strftime("%Y-%m-%d %H:%M"))
        start += timedelta(minutes=30)

    # GET RESERVED SLOTS FOR SAME DOCTOR
    cursor.execute(
        """
        SELECT scheduled_datetime
        FROM admitted_patients
        WHERE doctor_assigned = ?
        AND scheduled_datetime IS NOT NULL
        """,
        (doctor,)
    )

    reserved = cursor.fetchall()

    reserved_slots = [r[0] for r in reserved]

    conn.close()

    return render_template(
        "slots.html",
        id=id,
        all_slots=all_slots,
        reserved_slots=reserved_slots
    )


def parse_datetime_safe(dt_str):
    """
    Safely parse datetime strings in either '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d %H:%M'
    Returns a string in '%Y-%m-%d %H:%M' format.
    """
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(dt_str, fmt).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return dt_str  # fallback, just in case


def get_reserved_slots(hospital_name, selected_date):
    """Fetch reserved slots for a hospital on a given date."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT scheduled_datetime
        FROM admitted_patients
        WHERE emergency_contact=? AND scheduled_datetime LIKE ?
        """,
        (hospital_name, selected_date + "%")
    )
    reserved = cursor.fetchall()
    conn.close()

    # Convert to HH:MM strings
    return [datetime.strptime(r[0], "%Y-%m-%d %H:%M").strftime("%H:%M") for r in reserved]


@app.route("/hospital_slots", methods=["GET", "POST"])
def hospital_slots():
    hospital_name = ""
    selected_date = ""
    doctors_slots = {}  # key: doctor_name, value: {'slots': [], 'reserved': []}

    if request.method == "POST":
        hospital_name = request.form.get("hospital_name")
        selected_date = request.form.get("date")

        if hospital_name and selected_date:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()

            # Get all distinct doctors for this hospital on the selected date
            cursor.execute(
                """
                SELECT DISTINCT doctor_assigned
                FROM admitted_patients
                WHERE emergency_contact=? AND scheduled_datetime LIKE ?
                """,
                (hospital_name, selected_date + "%")
            )
            doctors = [r[0] for r in cursor.fetchall()]

            # Generate slots and reserved slots for each doctor
            for doctor in doctors:
                start = datetime.strptime(selected_date + " 09:00", "%Y-%m-%d %H:%M")
                end = datetime.strptime(selected_date + " 17:00", "%Y-%m-%d %H:%M")
                all_slots = []
                while start <= end:
                    all_slots.append(start.strftime("%H:%M"))
                    start += timedelta(minutes=30)

                # Fetch reserved slots for this doctor
                cursor.execute(
                    """
                    SELECT scheduled_datetime
                    FROM admitted_patients
                    WHERE emergency_contact=? AND doctor_assigned=? AND scheduled_datetime LIKE ?
                    """,
                    (hospital_name, doctor, selected_date + "%")
                )
                reserved = [r[0] for r in cursor.fetchall()]
                reserved_slots = [
                    datetime.strptime(r, "%Y-%m-%d %H:%M").strftime("%H:%M")
                    for r in reserved
                ]

                doctors_slots[doctor] = {"slots": all_slots, "reserved": reserved_slots}

            conn.close()

    return render_template(
        "doctor_slots.html",
        hospital_name=hospital_name,
        selected_date=selected_date,
        doctors_slots=doctors_slots
    )


# --------------------------
# Admin dashboard to view emergencies
# --------------------------
# --------------------------
# View Emergency Appointments
# --------------------------
@app.route("/admin/emergencies")
def view_emergencies():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # IMPORTANT
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM emergency_appointments ORDER BY createdatetime DESC")
    emergencies = cursor.fetchall()

    conn.close()

    return render_template("admin_emergencies.html", emergencies=emergencies)


# --------------------------
# Approve / Reject / Delete
# --------------------------
@app.route("/admin/emergency_action/<int:id>", methods=["POST"])
def emergency_action(id):
    action = request.form.get("action")

    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if action == "approve":

        cursor.execute(
            "UPDATE emergency_appointments SET status='Approved' WHERE id=?",
            (id,)
        )

        cursor.execute(
            "SELECT patient_name,mobile FROM emergency_appointments WHERE id=?",
            (id,)
        )

        row = cursor.fetchone()

        if row and row["mobile"]:
            send_confirmation_email(row["mobile"], row["patient_name"])

        flash("Emergency Approved & Email Sent", "success")


    elif action == "reject":

        cursor.execute(
            "UPDATE emergency_appointments SET status='Rejected' WHERE id=?",
            (id,)
        )

        flash("Emergency Rejected", "warning")


    elif action == "delete":

        cursor.execute(
            "DELETE FROM emergency_appointments WHERE id=?",
            (id,)
        )

        flash("Emergency Deleted", "danger")

    conn.commit()
    conn.close()

    return redirect(url_for("view_emergencies"))


# --------------------------
# Email Function
# --------------------------
def send_confirmation_email(to_email, patient_name):
    from_email = "vadde.seetha@gmail.com"
    password = "qoif fbmc fxst nrhp"

    subject = "Emergency Appointment Approved"

    body = f"""
Dear {patient_name},

Your emergency request has been approved.

Ambulance and hospital services are being arranged.

Hospital Emergency Team
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, password)
        server.send_message(msg)


# ==============================doctor_slots
# 🏥 Health Check
# ==============================

@app.route("/")
def home():
    return jsonify({"message": "Medical AI Backend Running Successfully 🚀"})


if __name__ == '__main__':
    app.run(debug=True)
