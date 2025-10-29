"""
AHA-EPM Cloud Deployment API
Deploys your trained model and Jarvis functions to the cloud.
Author: @gauravkumar2424
Date: 2025-01-29
"""

from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import openai
from gtts import gTTS
import os
import logging
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'MODEL_PATH': 'models_cloud/cloud_model.keras',
    'AUDIO_DIR': 'audio_output',
    'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
    'PORT': int(os.environ.get('PORT', 5000))
}

FEATURES = [
    'Vibration_RMS', 'Vibration_Peak', 'Vibration_Kurtosis', 'Vibration_PeakFreq',
    'Temperature_Mean', 'Temperature_Rate', 'Current_Variance', 'Current_SurgeCount',
    'Sound_Energy', 'Humidity_Mean', 'Humidity_Variance', 'TTF_Trend',
    'Vibration_Slope', 'Temperature_Slope', 'Vibration_Change', 'Is_Recovery',
    'temp_abnormal_duration'
]

FAULTS = [
    'Bearing Failure', 'Overheating', 'Load Imbalance', 'Blockage/Jamming',
    'Electrical Surges', 'Bearing', 'Recovery', 'Normal'
]

# ============================================================================
# FLASK SETUP
# ============================================================================
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if CONFIG['OPENAI_API_KEY']:
    openai.api_key = CONFIG['OPENAI_API_KEY']
    logger.info("OpenAI API key loaded from environment")
else:
    logger.warning("OpenAI API key not found in environment variables.")

os.makedirs(CONFIG['AUDIO_DIR'], exist_ok=True)

model = None

# ============================================================================
# MODEL AND UTILITY FUNCTIONS
# ============================================================================

def asymmetric_ttf_loss(y_true, y_pred):
    error = y_pred - y_true
    loss = tf.where(error > 0, error**2 * 3.0, error**2)
    return tf.reduce_mean(loss)

def load_model():
    global model
    try:
        logger.info("Loading trained model...")
        model = tf.keras.models.load_model(
            CONFIG['MODEL_PATH'],
            custom_objects={'asymmetric_ttf_loss': asymmetric_ttf_loss}
        )
        logger.info("Model loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return False

def get_jarvis_analysis(fault_probabilities, ttf_prediction, sensor_features, edge_prediction=None):
    fault_data = "FAULT PROBABILITY ANALYSIS:\n"
    for i, fault in enumerate(FAULTS):
        if fault_probabilities[i] > 0.05:
            fault_data += f"{fault}: {fault_probabilities[i]*100:.2f}%\n"

    sensor_data = "SENSOR READINGS:\n"
    critical_features = ['Vibration_RMS', 'Temperature_Mean', 'Current_Variance', 'Sound_Energy', 'TTF_Trend']
    for i, feature in enumerate(FEATURES):
        if feature in critical_features:
            sensor_data += f"{feature}: {sensor_features[i]:.4f}\n"

    edge_context = ""
    if edge_prediction:
        edge_context = (
            "EDGE DEVICE PREDICTION:\n"
            f"Edge TTF: {edge_prediction.get('ttf', 'N/A')} hours\n"
            f"Edge Confidence: {edge_prediction.get('confidence', 'N/A')}\n"
            f"Edge Top Fault: {edge_prediction.get('top_fault', 'N/A')}\n"
        )

    prompt = (
        "You are Jarvis, an expert AI system for industrial predictive maintenance. "
        "Analyze this equipment data using your expertise.\n\n"
        f"{fault_data}\n"
        f"PREDICTED TIME TO FAILURE: {ttf_prediction:.3f} hours ({ttf_prediction*60:.1f} minutes)\n\n"
        f"{sensor_data}\n\n"
        f"{edge_context}\n"
        "TASK:\n"
        "Analyze this data as an expert maintenance engineer would. Use your intelligence to:\n"
        "1. Determine the actual equipment health status by interpreting all the probabilities together\n"
        "2. Identify the primary failure mode and explain why (based on sensor correlations)\n"
        "3. Explain how multiple faults interact or compound each other\n"
        "4. Provide specific maintenance actions in priority order\n"
        "5. Estimate maintenance time and criticality\n"
        "6. Suggest monitoring parameters to track\n"
        "Be concise, actionable, and professional."
    )

    try:
        if not CONFIG['OPENAI_API_KEY']:
            raise Exception("OpenAI API key not configured")

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are Jarvis, an expert AI predictive maintenance system. You have deep knowledge of mechanical engineering, failure analysis, and industrial equipment. Provide expert-level analysis and recommendations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=400,
            temperature=0.3
        )

        jarvis_analysis = response.choices[0].message.content
        logger.info("Jarvis AI analysis complete")
        return jarvis_analysis

    except Exception as e:
        logger.error(f"Jarvis AI error: {e}")
        top_fault_idx = np.argmax(fault_probabilities)
        return (
            "Analysis unavailable (Jarvis AI offline).\n\n"
            f"RAW DATA:\nTime to Failure: {ttf_prediction:.2f} hours ({ttf_prediction*60:.0f} minutes)\n"
            f"Top Fault Probabilities: {', '.join([f'{FAULTS[i]}:{fault_probabilities[i]*100:.0f}%' for i in np.argsort(fault_probabilities)[-3:][::-1]])}\n"
            f"Primary Fault: {FAULTS[top_fault_idx]} ({fault_probabilities[top_fault_idx]*100:.0f}%)\n"
            "Please consult maintenance team for detailed interpretation."
        )

def generate_jarvis_speech(text, device_id="esp32"):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jarvis_{device_id}_{timestamp}.mp3"
        filepath = os.path.join(CONFIG['AUDIO_DIR'], filename)

        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(filepath)

        logger.info(f"Jarvis speech generated: {filename} ({os.path.getsize(filepath)/1024:.1f} KB)")
        return filename
    except Exception as e:
        logger.error(f"Speech generation error: {e}")
        return None

def fuse_edge_cloud_predictions(edge_class, edge_ttf, edge_conf, cloud_class, cloud_ttf):
    alpha = edge_conf
    fused_class = alpha * np.array(edge_class) + (1 - alpha) * np.array(cloud_class)
    fused_ttf = alpha * edge_ttf + (1 - alpha) * cloud_ttf
    return fused_class.tolist(), float(fused_ttf), float(alpha)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    return jsonify({
        'status': 'online',
        'service': 'AHA-EPM Cloud API',
        'version': '1.0.0',
        'author': '@gauravkumar2424',
        'date': '2025-01-29',
        'model_loaded': model is not None,
        'openai_configured': CONFIG['OPENAI_API_KEY'] is not None,
        'endpoints': {
            'predict': '/predict (POST)',
            'audio': '/audio/<filename> (GET)',
            'health': '/health (GET)'
        }
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'openai_configured': CONFIG['OPENAI_API_KEY'] is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data received'}), 400

        device_id = data.get('device_id', 'ESP32_UNKNOWN')
        sensor_data = data.get('sensor_data', [])
        edge_prediction = data.get('edge_prediction', None)

        logger.info("="*70)
        logger.info(f"Request from: {device_id}")
        logger.info("="*70)

        if len(sensor_data) != 17:
            return jsonify({'error': 'Expected 17 sensor features'}), 400

        X = np.array(sensor_data).reshape(1, -1)
        logger.info("Running model prediction...")
        y_class, y_ttf, y_diag = model.predict(X, verbose=0)

        cloud_class_probs = y_class[0]
        cloud_ttf = float(y_ttf[0][0])

        logger.info(f"Cloud TTF: {cloud_ttf:.2f}h")
        logger.info(f"Top fault: {FAULTS[np.argmax(cloud_class_probs)]} ({max(cloud_class_probs)*100:.1f}%)")

        if edge_prediction and 'class_probs' in edge_prediction:
            logger.info("Fusing edge and cloud predictions...")
            fused_class, fused_ttf, fusion_weight = fuse_edge_cloud_predictions(
                edge_prediction['class_probs'],
                edge_prediction.get('ttf', cloud_ttf),
                edge_prediction.get('confidence', 0.5),
                cloud_class_probs.tolist(),
                cloud_ttf
            )
            final_class_probs = fused_class
            final_ttf = fused_ttf
        else:
            logger.info("Using cloud-only predictions")
            final_class_probs = cloud_class_probs.tolist()
            final_ttf = cloud_ttf
            fusion_weight = 0.0

        logger.info("Calling Jarvis AI analysis...")
        jarvis_text = get_jarvis_analysis(
            fault_probabilities=np.array(final_class_probs),
            ttf_prediction=final_ttf,
            sensor_features=sensor_data,
            edge_prediction=edge_prediction
        )

        logger.info("Generating speech...")
        audio_filename = generate_jarvis_speech(jarvis_text, device_id)

        if audio_filename:
            audio_url = url_for('get_audio', filename=audio_filename, _external=True)
        else:
            audio_url = None

        response = {
            'success': True,
            'device_id': device_id,
            'timestamp': datetime.now().isoformat(),
            'predictions': {
                'fault_probabilities': {FAULTS[i]: final_class_probs[i] for i in range(8)},
                'ttf_hours': final_ttf,
                'ttf_minutes': final_ttf * 60,
                'primary_fault': FAULTS[np.argmax(final_class_probs)],
                'primary_fault_probability': float(max(final_class_probs)),
                'diagnostics': y_diag[0].tolist(),
                'fusion_weight': fusion_weight
            },
            'jarvis': {
                'analysis': jarvis_text,
                'audio_url': audio_url,
                'audio_filename': audio_filename
            }
        }

        logger.info("Prediction complete.")
        logger.info("="*70)

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    try:
        filepath = os.path.join(CONFIG['AUDIO_DIR'], filename)

        if not os.path.exists(filepath):
            return jsonify({'error': 'Audio not found'}), 404

        logger.info(f"Serving audio: {filename}")
        return send_file(filepath, mimetype='audio/mpeg')

    except Exception as e:
        logger.error(f"Audio error: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================================
# STARTUP - ENSURE MODEL LOADS ON IMPORT (FOR GUNICORN!)
# ============================================================================

if not load_model():
    logger.error("Failed to load model. Exiting.")
    raise RuntimeError("Failed to load model.")

logger.info("API ready to receive ESP32 requests.")

if __name__ == '__main__':
    logger.info("="*70)
    logger.info("AHA-EPM CLOUD API STARTING")
    logger.info("="*70)
    logger.info(f"Author: @gauravkumar2424")
    logger.info(f"Date: 2025-01-29")
    logger.info(f"Port: {CONFIG['PORT']}")
    logger.info("="*70)
    app.run(host='0.0.0.0', port=CONFIG['PORT'], debug=False)
