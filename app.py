import os
import cv2
import pymysql
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from helmet_detection import detect_helmet, detect_people  # ✅ Correct function name
from one_way_detection import predict_image  # ✅ Import One-Way Detection
from triple_seat_detection import predict_image as predict_triple_seat
from reportlab.pdfgen import canvas

app = Flask(__name__)

# ✅ MySQL Configuration
db = pymysql.connect(
    host="localhost",
    user="root",
    password="",
    database="detection_system",
    cursorclass=pymysql.cursors.DictCursor
)
cursor = db.cursor()

# ✅ Set upload & result folders
UPLOAD_FOLDER = "static/uploads/"
RESULT_FOLDER = "static/results/"
CHALLAN_FOLDER = "static/challans/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["CHALLAN_FOLDER"] = CHALLAN_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(CHALLAN_FOLDER, exist_ok=True)

def generate_challan(filename, violation_type):
    challan_folder = app.config["CHALLAN_FOLDER"]
    
    # ✅ Debugging print
    print(f"🟢 generate_challan() called with filename: {filename}, violation: {violation_type}")
    
    # ✅ Ensure folder exists
    if not os.path.exists(challan_folder):
        print(f"📁 Creating folder: {challan_folder}")
        os.makedirs(challan_folder, exist_ok=True)

    pdf_path = os.path.join(challan_folder, filename)
    print(f"📄 Attempting to save Challan at: {pdf_path}")

    try:
        c = canvas.Canvas(pdf_path)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Traffic Violation Challan")

        c.setFont("Helvetica", 12)
        c.drawString(100, 700, f"Violation: {violation_type}")
        c.drawString(100, 680, "Fine Amount: ₹1000")

        c.drawString(100, 650, "Issued By: Traffic Authority")
        c.drawString(100, 630, "Date: __/__/2025")

        c.save()
        print(f"✅ Challan successfully saved at {pdf_path}")  # Debugging print
        return filename
    except Exception as e:
        print(f"🚨 Error while generating challan: {e}")
        return None

# ✅ Dashboard Route (First Page)
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/helmet", methods=["GET", "POST"])
def helmet_detection():
    result_image = None
    message = "Upload an image to check for helmet violations."
    challan_pdf = None  # 🆕 To store challan filename
    file_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file!", 400

        # ✅ Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # ✅ Run YOLO detection
        result_path, detections = detect_helmet(file_path)

        # ✅ Check if violation exists
        if any(label == "Without Helmet" for label, _ in detections):
            message = "🚨 No Helmet Detected!"
            challan_filename = f"challan_{filename}.pdf"
            print(f"📄 Generating Challan: {challan_filename}")  # Debugging
            challan_pdf = generate_challan(challan_filename, "Helmet Violation")
            print(f"📄 Challan PDF Path: {challan_pdf}")  # Debugging
        else:
            message = "✅ Helmet Detected."

        result_image = result_path

    return render_template("index.html", result_image=result_image, message=message, challan_pdf=challan_pdf)

@app.route("/multi_rider", methods=["POST", "GET"])
def multi_rider_detection():
    result_image = None
    message = ""

    if request.method == "POST":
        print("✅ POST request received")  # Debugging step

        if "file" not in request.files:
            print("🚨 No file part in request!")  # Debugging step
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            print("🚨 No selected file!")  # Debugging step
            return "No selected file!", 400

        print(f"✅ File received: {file.filename}")  # Debugging step

        # ✅ Save uploaded file correctly
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        print(f"✅ File saved at: {file_path}")  # Debugging step

        # ✅ Detect multiple riders
        result_path, message = detect_people(file_path)  

        print(f"✅ Detection done. Result at: {result_path}")  # Debugging step

        result_image = result_path  

    return render_template("multi_rider_detection.html", result_image=result_image, message=message)

@app.route("/one_way", methods=["POST", "GET"])
def one_way_detection():
    result_image = None
    message = ""

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file!", 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # ✅ Call prediction function
        prediction, probability = predict_image(file_path)

        if prediction is None or probability is None:
            return "Error processing image!", 500  # ❌ Handle None case

        message = f"Detected: {prediction} (Confidence: {probability:.2f})"
        result_image = file.filename  # ✅ Save result image filename

    return render_template("one_way_detection.html", result_image=result_image, message=message)

@app.route("/triple_seat", methods=["POST", "GET"])
def triple_seat_detection():
    violation_label = None
    result_image = None
    message = "Upload an image to check for violations."
    challan_pdf = None
    file_path = None

    if request.method == "POST":
        print("✅ POST request received for Triple Seat Detection")  # Debugging

        if "file" not in request.files:
            print("🚨 No file uploaded!")
            return "No file uploaded!", 400

        file = request.files["file"]
        if file.filename == "":
            print("🚨 No selected file!")
            return "No selected file!", 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        print(f"✅ Image saved at {file_path}")  # Debugging

        # ✅ Get only the prediction label
        violation_label, _ = predict_triple_seat(file_path)  

        if violation_label.lower() == "triple_seat":  # ✅ Case-insensitive check
            print("🚨 Triple Seat Violation Detected!")
            message = "🚨 Triple Seat Violation Detected!"
            challan_filename = f"challan_{filename}.pdf"
            print(f"📄 Generating Challan: {challan_filename}")  # Debugging
            challan_pdf = generate_challan(challan_filename, "Triple Seat Violation")
            print(f"📄 Challan PDF Path: {challan_pdf}")  # Debugging
        else:
            print("✅ No Violation Detected.")
            message = "✅ No Violation Detected."

        result_image = filename

    return render_template(
        "triple_seat_detection.html",
        violation_label=violation_label,
        result_image=result_image,
        message=message,
        image_path=file_path if file_path else "",
        challan_pdf=challan_pdf
    )

# ✅ Route to Download Challan PDF
@app.route("/download_challan/<filename>")
def download_challan(filename):
    challan_path = os.path.join(app.config["CHALLAN_FOLDER"], filename)
    return send_file(challan_path, as_attachment=True)

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        if name and email and message:
            try:
                sql = "INSERT INTO feedback (name, email, message) VALUES (%s, %s, %s)"
                cursor.execute(sql, (name, email, message))
                db.commit()
                return "Feedback submitted successfully!"
            except Exception as e:
                return f"Error: {e}"
        else:
            return "All fields are required!", 400

    return render_template("feedback.html")  # Create a feedback.html template

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # 🔥 Disable auto-reload
