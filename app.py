"""
Plant Disease Detection System - Flask Web Application
======================================================
Serves the frontend and provides the prediction API.

Usage:
    python app.py

The server starts on http://localhost:5000
"""

import os
import uuid
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from predict import (load_model, load_class_names, preprocess_image, predict,
                      get_disease_info, is_prediction_reliable, predict_with_gemini)

# =============================================================================
# App Configuration
# =============================================================================
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload size
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Global model and class names — loaded once at startup
model = None
class_names = None


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def init_model():
    """Load the trained model and class names at startup."""
    global model, class_names

    model_path = os.path.join('models', 'plant_disease_model.keras')
    class_path = os.path.join('models', 'class_names.json')

    print("\n" + "=" * 60)
    print("  Plant Disease Detection System — Starting Up")
    print("=" * 60)

    model = load_model(model_path)
    class_names = load_class_names(class_path)

    if model is not None and class_names is not None:
        print(f"\n  [OK] Model loaded successfully ({len(class_names)} classes)")
        print("  [OK] Ready for predictions")
    else:
        print("\n  [WARNING] Model NOT loaded -- predictions will be unavailable.")
        print("  -> Train the model first: python train_model.py")
        print("  -> The web interface will still load, but /predict will return an error.")

    print("=" * 60 + "\n")


# =============================================================================
# Routes
# =============================================================================

@app.before_request
def ensure_model_loaded():
    """Ensure model is loaded on serverless cold starts (e.g. Vercel)."""
    global model, class_names
    if model is None:
        init_model()


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files explicitly in serverless environments like Vercel."""
    static_dir = os.path.join(app.root_path, 'static')
    return send_from_directory(static_dir, filename)


@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_disease():
    """
    Receive an uploaded leaf image and return disease prediction.

    Returns JSON:
        {
            "success": true,
            "predictions": [
                {
                    "class_name": "Tomato___Early_blight",
                    "confidence": 92.3,
                    "is_healthy": false,
                    "plant_name": "Tomato",
                    "disease_name": "Early blight"
                },
                ...
            ],
            "disease_info": {
                "description": "...",
                "prevention": ["..."]
            },
            "image_url": "/static/uploads/abc123.jpg"
        }
    """
    # Check if model is loaded
    if model is None or class_names is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first by running: python train_model.py'
        }), 503

    # Check if file is present in request
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file uploaded. Please select an image file.'
        }), 400

    file = request.files['file']

    # Check if filename is not empty
    if file.filename == '' or file.filename is None:
        return jsonify({
            'success': False,
            'error': 'No file selected. Please choose an image to upload.'
        }), 400

    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': 'Invalid file type. Please upload a JPG, JPEG, or PNG image.'
        }), 400

    try:
        # Read uploaded image bytes directly
        img_bytes = file.read()

        # Build base64 Data URL (serverless/Vercel compatible, no read-only disk errors)
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'jpeg'
        mime = 'image/png' if ext == 'png' else 'image/jpeg'
        b64_str = base64.b64encode(img_bytes).decode('utf-8')
        image_url = f"data:{mime};base64,{b64_str}"

        # Try saving local copy if upload directory is writable
        try:
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
        except Exception:
            pass  # Expected on serverless platforms (e.g. Vercel)

        # Preprocess directly from bytes in memory
        image_array = preprocess_image(img_bytes)
        if image_array is None:
            return jsonify({
                'success': False,
                'error': 'Invalid or corrupted image. Please upload a valid leaf photo.'
            }), 400

        # Check for Gemini Vision API key (via custom header or server env)
        gemini_key = request.headers.get('X-Gemini-Key') or os.environ.get('GEMINI_API_KEY')
        gemini_result = None

        if gemini_key:
            gemini_result = predict_with_gemini(img_bytes, gemini_key.strip())

        if gemini_result and gemini_result.get('predictions'):
            predictions = gemini_result['predictions']
            disease_info = gemini_result.get('disease_info') or get_disease_info(predictions[0].get('class_name', ''))
            engine_name = gemini_result.get('engine', 'Google Gemini Vision AI')
            is_reliable = True
            warning = None
        else:
            # Run local neural & agronomic multi-crop prediction
            predictions = predict(model, image_array, class_names, top_k=3, raw_bytes=img_bytes)

            if not predictions:
                return jsonify({
                    'success': False,
                    'error': 'Prediction failed. The model could not classify this image.'
                }), 500

            reliability = is_prediction_reliable(predictions)
            top_class = predictions[0].get('class_name', '')
            disease_info = get_disease_info(top_class)
            engine_name = "Agronomic Vision AI (Rice Engine)" if "Rice" in top_class else "MobileNetV2 Vision AI"
            is_reliable = reliability['reliable']
            warning = reliability.get('reason')

        # Build final response
        response = {
            'success': True,
            'predictions': predictions,
            'disease_info': disease_info,
            'image_url': image_url,
            'is_reliable': is_reliable,
            'engine': engine_name,
        }

        if warning:
            response['warning'] = warning

        return jsonify(response), 200

    except Exception as e:
        print(f"[ERROR] Prediction request failed: {e}")
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred while processing the image. Please try again.'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    Returns JSON:
        {
            "status": "ok",
            "model_loaded": true,
            "classes": 38
        }
    """
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'classes': len(class_names) if class_names else 0
    }), 200


@app.errorhandler(413)
def file_too_large(e):
    """Handle file size exceeding limit."""
    return jsonify({
        'success': False,
        'error': 'File is too large. Maximum allowed size is 10 MB.'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'The requested resource was not found.'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({
        'success': False,
        'error': 'An internal server error occurred. Please try again.'
    }), 500


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Load model at startup
    init_model()

    # Start Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)
