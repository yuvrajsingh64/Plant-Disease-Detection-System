"""
Plant Disease Detection System - Prediction Module
===================================================
Handles model loading, image preprocessing, and disease prediction.
Used by app.py for inference — never trains the model.
"""

import os
import json
import numpy as np
from typing import Optional, Union
from PIL import Image
import io


def load_model(model_path: str = os.path.join('models', 'plant_disease_model.keras')):
    """
    Load the trained model from disk.
    Supports both Keras (.keras) models and lightweight TensorFlow Lite (.tflite)
    models for cloud/serverless environments like Vercel.

    Args:
        model_path: Path to the saved model file.

    Returns:
        Loaded model/interpreter, or None if loading fails.
    """
    # 1. First check if Keras model exists and can be loaded
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"[INFO] Keras model loaded successfully from: {model_path}")
            return model
        except ImportError:
            print("[INFO] Full TensorFlow not available, falling back to TFLite...")
        except Exception as e:
            print(f"[WARNING] Could not load Keras model: {e}")

    # 2. Check for TFLite model (.tflite) - ideal for Vercel/serverless
    tflite_path = os.path.join('models', 'plant_disease_model.tflite')
    if os.path.exists(tflite_path):
        try:
            # Try lightweight tflite-runtime or ai_edge_litert first
            try:
                import tflite_runtime.interpreter as tflite
            except ImportError:
                try:
                    from ai_edge_litert import interpreter as tflite
                except ImportError:
                    import tensorflow.lite as tflite

            interpreter = tflite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            print(f"[INFO] TFLite model loaded successfully from: {tflite_path}")
            return interpreter
        except Exception as e:
            print(f"[ERROR] Failed to load TFLite model: {e}")

    print(f"[WARNING] Model file not found at {model_path} or {tflite_path}")
    print("  Please train the model first: python train_model.py")
    return None


def load_class_names(json_path: str = os.path.join('models', 'class_names.json')) -> Optional[list]:
    """
    Load class name mapping from JSON file.

    Args:
        json_path: Path to the class_names.json file.

    Returns:
        List of class name strings, or None if loading fails.
    """
    if not os.path.exists(json_path):
        print(f"[WARNING] Class names file not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            class_names = json.load(f)
        print(f"[INFO] Loaded {len(class_names)} class names from: {json_path}")
        return class_names
    except Exception as e:
        print(f"[ERROR] Failed to load class names: {e}")
        return None


def preprocess_image(image_source: Union[str, bytes]) -> Optional[np.ndarray]:
    """
    Preprocess an image for model prediction.

    Accepts a file path (str) or raw image bytes. Validates the image,
    converts to RGB, resizes to 224x224, normalizes to [0, 1], and
    adds a batch dimension.

    Args:
        image_source: File path string or raw image bytes.

    Returns:
        Numpy array of shape (1, 224, 224, 3) with values in [0, 1],
        or None if preprocessing fails.
    """
    IMG_SIZE = (224, 224)

    try:
        # Open the image
        if isinstance(image_source, str):
            if not os.path.exists(image_source):
                print(f"[ERROR] Image file not found: {image_source}")
                return None
            img = Image.open(image_source)
        elif isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source))
        else:
            print(f"[ERROR] Invalid image source type: {type(image_source)}")
            return None

        # Validate it's a real image by loading pixel data
        img.verify()
        # Re-open after verify (verify() can only be called once)
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = Image.open(io.BytesIO(image_source))

        # Convert to RGB (handles RGBA, grayscale, palette, etc.)
        img = img.convert('RGB')

        # Resize to model input size
        img = img.resize(IMG_SIZE, Image.LANCZOS)

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except (IOError, OSError, SyntaxError) as e:
        print(f"[ERROR] Invalid or corrupted image: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        return None


def parse_class_name(class_name: str) -> dict:
    """
    Parse a PlantVillage-style class name into plant and disease components.

    Class names follow the format: 'Plant___Disease_Name'
    Example: 'Tomato___Early_blight' -> {'plant': 'Tomato', 'disease': 'Early blight'}

    Args:
        class_name: Raw class name string.

    Returns:
        Dict with 'plant_name', 'disease_name', and 'is_healthy' keys.
    """
    is_healthy = 'healthy' in class_name.lower()

    if '___' in class_name:
        parts = class_name.split('___', 1)
        plant_name = parts[0].replace('_', ' ').strip()
        disease_name = parts[1].replace('_', ' ').strip()
    else:
        plant_name = class_name.replace('_', ' ').strip()
        disease_name = 'Healthy' if is_healthy else 'Unknown'

    if is_healthy:
        disease_name = 'Healthy'

    return {
        'plant_name': plant_name,
        'disease_name': disease_name,
        'is_healthy': is_healthy
    }

# List of plant species supported by the trained model
SUPPORTED_PLANTS = [
    'Apple', 'Blueberry', 'Cherry', 'Corn (maize)', 'Grape', 'Orange',
    'Peach', 'Pepper (bell)', 'Potato', 'Raspberry', 'Soybean',
    'Squash', 'Strawberry', 'Tomato'
]

MIN_CONFIDENCE_THRESHOLD = 55.0


def is_prediction_reliable(predictions: list) -> dict:
    """
    Evaluate whether the top prediction is reliable and whether the uploaded
    plant is likely supported by the model.

    Returns:
        dict: {
            'reliable': bool,
            'reason': str or None,
            'warning_type': 'unsupported_plant' | 'low_confidence' | 'ambiguous' | None
        }
    """
    if not predictions:
        return {
            'reliable': False,
            'reason': 'No prediction could be made for this image.',
            'warning_type': 'no_prediction'
        }

    top = predictions[0]
    top_conf = top['confidence']

    # Severe low confidence (< 50%): almost certainly an unsupported plant (like Paddy/Rice, Wheat, Mango, etc.) or non-leaf
    if top_conf < 50.0:
        return {
            'reliable': False,
            'warning_type': 'unsupported_plant',
            'reason': (
                f"⚠️ Inconclusive Diagnosis ({top_conf:.1f}% confidence): The uploaded image does not strongly match any supported crop in the database. "
                "If this is a Paddy/Rice, Wheat, or other unlisted plant, please note that this system currently supports: "
                f"{', '.join(SUPPORTED_PLANTS)}. Please ensure you upload a clear close-up leaf photo of a supported crop."
            )
        }

    # Moderate confidence (< 65%) with close runner-up (gap < 10%): ambiguous
    if len(predictions) > 1:
        gap = top_conf - predictions[1]['confidence']
        if top_conf < 65.0 and gap < 10.0:
            return {
                'reliable': False,
                'warning_type': 'ambiguous',
                'reason': (
                    f"⚠️ Uncertain Diagnosis ({top_conf:.1f}% vs {predictions[1]['confidence']:.1f}%): The AI is torn between '{top['plant_name']} - {top['disease_name']}' "
                    f"and '{predictions[1]['plant_name']} - {predictions[1]['disease_name']}'. This often happens when the leaf belongs to an unsupported plant type (e.g. Paddy/Rice) "
                    "or the photo is taken from too far away."
                )
            }

    # Moderate confidence warning
    if top_conf < 70.0:
        return {
            'reliable': True,
            'warning_type': 'moderate_confidence',
            'reason': (
                f"ℹ️ Moderate confidence ({top_conf:.1f}%). For highest accuracy, capture a well-lit, sharp photo focused on an individual leaf."
            )
        }

    return {'reliable': True, 'reason': None, 'warning_type': None}


def predict(model, image_array: np.ndarray, class_names: list, top_k: int = 3) -> list:
    """
    Run inference and return top-k predictions.

    Args:
        model: Loaded Keras model.
        image_array: Preprocessed image array of shape (1, 224, 224, 3).
        class_names: List of class name strings.
        top_k: Number of top predictions to return.

    Returns:
        List of dicts, each with keys: 'class_name', 'confidence',
        'is_healthy', 'plant_name', 'disease_name'.
    """
    if model is None:
        raise ValueError("Model is not loaded.")
    if image_array is None:
        raise ValueError("Image array is None.")
    if class_names is None or len(class_names) == 0:
        raise ValueError("Class names list is empty.")

    # Run prediction (handles both Keras model and TFLite interpreter)
    if hasattr(model, 'get_input_details'):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], image_array)
        model.invoke()
        probabilities = model.get_tensor(output_details[0]['index'])[0]
    else:
        predictions = model.predict(image_array, verbose=0)
        probabilities = predictions[0]  # Shape: (num_classes,)

    # Get top-k indices
    top_k = min(top_k, len(class_names))
    top_indices = np.argsort(probabilities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        class_name = class_names[idx]
        confidence = float(probabilities[idx]) * 100  # Convert to percentage
        parsed = parse_class_name(class_name)

        results.append({
            'class_name': class_name,
            'confidence': round(confidence, 2),
            'is_healthy': parsed['is_healthy'],
            'plant_name': parsed['plant_name'],
            'disease_name': parsed['disease_name']
        })

    return results


# =============================================================================
# Disease Information Database
# =============================================================================
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'description': 'Apple scab is a fungal disease caused by Venturia inaequalis. It appears as olive-green to dark brown lesions on leaves and fruit. Severe infections can cause leaf drop and reduce fruit quality.',
        'prevention': [
            'Plant scab-resistant apple varieties when possible.',
            'Apply fungicide sprays during the primary infection period (spring).',
            'Rake and destroy fallen leaves in autumn to reduce overwintering spores.',
            'Ensure good air circulation by proper pruning.',
            'Avoid overhead irrigation to keep foliage dry.'
        ]
    },
    'Apple___Black_rot': {
        'description': 'Black rot is caused by the fungus Botryosphaeria obtusa. It produces circular brown lesions on leaves with concentric rings and causes fruit rot with black, mummified fruit.',
        'prevention': [
            'Remove and destroy infected fruit, dead wood, and cankers.',
            'Prune trees to improve air circulation.',
            'Apply appropriate fungicides during the growing season.',
            'Maintain tree health through proper fertilization.',
            'Remove mummified fruit from trees and ground.'
        ]
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae. It requires both cedar/juniper and apple hosts. Bright orange-yellow spots appear on apple leaves.',
        'prevention': [
            'Remove nearby cedar or juniper trees if practical.',
            'Plant rust-resistant apple varieties.',
            'Apply protective fungicides in spring.',
            'Monitor for early symptoms and remove infected leaves.',
            'Ensure good air circulation around trees.'
        ]
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Gray leaf spot is a fungal disease caused by Cercospora zeae-maydis. It produces rectangular gray-brown lesions on corn leaves, which can significantly reduce yield in severe cases.',
        'prevention': [
            'Rotate crops to reduce disease pressure.',
            'Plant resistant corn hybrids.',
            'Apply foliar fungicides if disease pressure is high.',
            'Manage crop residue through tillage.',
            'Ensure adequate plant spacing for air flow.'
        ]
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Common rust of corn is caused by Puccinia sorghi. Small, circular to elongated reddish-brown pustules appear on both leaf surfaces. It is generally a minor disease but can reduce yield in susceptible varieties.',
        'prevention': [
            'Plant resistant corn hybrids.',
            'Apply foliar fungicides if rust appears early.',
            'Monitor fields regularly for early detection.',
            'Remove volunteer corn plants that harbor rust.',
            'Balanced fertilization helps plants resist infection.'
        ]
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Northern leaf blight is caused by Exserohilum turcicum. It produces long, cigar-shaped gray-green lesions on corn leaves that can expand and merge, reducing photosynthetic capacity.',
        'prevention': [
            'Plant resistant corn hybrids.',
            'Rotate corn with non-host crops.',
            'Apply foliar fungicides at early disease stages.',
            'Manage crop residue to reduce inoculum.',
            'Maintain balanced nutrition for plant health.'
        ]
    },
    'Grape___Black_rot': {
        'description': 'Grape black rot is caused by the fungus Guignardia bidwellii. It produces brown circular lesions on leaves and causes grapes to shrivel into hard, dark mummies.',
        'prevention': [
            'Remove mummified berries and infected canes during pruning.',
            'Apply fungicides from bud break through veraison.',
            'Maintain good canopy management and air circulation.',
            'Remove wild grapes near vineyards.',
            'Practice good sanitation in the vineyard.'
        ]
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Esca (black measles) is a complex grapevine trunk disease caused by several fungi. It produces interveinal discoloration and necrosis, often called "tiger stripe" patterns on leaves.',
        'prevention': [
            'Protect pruning wounds with sealants or fungicides.',
            'Avoid large pruning wounds when possible.',
            'Remove and destroy severely infected vines.',
            'Double-prune to delay wound exposure in wet weather.',
            'No fully effective cure exists; manage through prevention.'
        ]
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Grape leaf blight (Isariopsis leaf spot) is caused by Pseudocercospora vitis. It produces dark brown spots with yellow halos on leaves, leading to premature defoliation.',
        'prevention': [
            'Apply appropriate fungicides during the growing season.',
            'Improve air circulation through canopy management.',
            'Remove fallen infected leaves.',
            'Avoid overhead watering.',
            'Maintain vine health through balanced nutrition.'
        ]
    },
    'Potato___Early_blight': {
        'description': 'Early blight of potato is caused by Alternaria solani. It produces dark brown spots with concentric rings (target-like appearance) on older leaves first, progressing upward.',
        'prevention': [
            'Rotate crops on a 2-3 year cycle.',
            'Plant certified disease-free seed potatoes.',
            'Apply fungicides preventatively before symptoms appear.',
            'Remove and destroy crop debris after harvest.',
            'Ensure adequate plant spacing and ventilation.'
        ]
    },
    'Potato___Late_blight': {
        'description': 'Late blight is caused by the oomycete Phytophthora infestans. It produces water-soaked lesions that quickly turn brown-black. This is the same disease that caused the Irish Potato Famine.',
        'prevention': [
            'Plant resistant varieties when available.',
            'Apply protective fungicides preventatively.',
            'Destroy cull piles and volunteer potatoes.',
            'Avoid overhead irrigation.',
            'Harvest promptly and store tubers in cool, dry conditions.'
        ]
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial spot is caused by Xanthomonas species. It produces small, dark, water-soaked lesions on leaves, stems, and fruit. Severe infections can cause significant defoliation and fruit loss.',
        'prevention': [
            'Use disease-free seeds and transplants.',
            'Apply copper-based bactericides preventatively.',
            'Avoid working with plants when they are wet.',
            'Rotate crops and avoid planting in previously infected areas.',
            'Remove and destroy infected plant material.'
        ]
    },
    'Tomato___Early_blight': {
        'description': 'Tomato early blight is caused by Alternaria solani. It produces dark concentric spots (target pattern) on lower leaves first, then progresses upward. Causes leaf yellowing and drop.',
        'prevention': [
            'Rotate crops annually.',
            'Mulch around plants to prevent soil splash.',
            'Apply fungicides preventatively.',
            'Water at the base of plants, not overhead.',
            'Remove lower leaves that touch the soil.'
        ]
    },
    'Tomato___Late_blight': {
        'description': 'Tomato late blight is caused by Phytophthora infestans. It produces large, irregularly shaped, water-soaked gray-green areas on leaves that quickly turn brown. White fuzzy growth may appear on the underside.',
        'prevention': [
            'Plant resistant varieties.',
            'Apply fungicides before symptoms appear in wet conditions.',
            'Improve air circulation through proper spacing and staking.',
            'Remove and destroy infected plants immediately.',
            'Water at the base; avoid wetting foliage.'
        ]
    },
    'Tomato___Leaf_Mold': {
        'description': 'Leaf mold is caused by the fungus Passalora fulva. It produces pale green to yellow spots on upper leaf surfaces and olive-green to brown velvety growth on the undersides.',
        'prevention': [
            'Improve air circulation in greenhouses and fields.',
            'Reduce humidity by ventilating growing areas.',
            'Apply fungicides when conditions favor disease.',
            'Plant resistant varieties.',
            'Avoid overhead irrigation.'
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Septoria leaf spot is caused by Septoria lycopersici. It produces small, circular spots with dark borders and gray centers containing tiny dark dots (pycnidia).',
        'prevention': [
            'Remove and destroy lower infected leaves.',
            'Apply fungicides at first sign of disease.',
            'Rotate with non-solanaceous crops for 2-3 years.',
            'Stake and mulch plants to reduce soil splash.',
            'Avoid overhead watering.'
        ]
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Two-spotted spider mites (Tetranychus urticae) are tiny arachnids that feed on leaf cells, causing stippling, yellowing, and bronzing. Heavy infestations produce visible webbing.',
        'prevention': [
            'Monitor plants regularly with a hand lens.',
            'Encourage natural predators like ladybugs and lacewings.',
            'Apply miticides or insecticidal soap when populations are high.',
            'Increase humidity around plants (mites prefer dry conditions).',
            'Remove heavily infested leaves or plants.'
        ]
    },
    'Tomato___Target_Spot': {
        'description': 'Target spot is caused by Corynespora cassiicola. It produces small, dark brown circular spots with concentric rings (target pattern) on leaves, stems, and fruit.',
        'prevention': [
            'Apply fungicides preventatively.',
            'Rotate crops with non-host plants.',
            'Improve air circulation through proper spacing.',
            'Remove and destroy crop debris after harvest.',
            'Avoid overhead watering.'
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Tomato yellow leaf curl virus (TYLCV) is transmitted by whiteflies. It causes severe leaf curling, yellowing, stunting, and significant yield loss. There is no cure once infected.',
        'prevention': [
            'Control whitefly populations with insecticides or biological control.',
            'Use reflective mulches to repel whiteflies.',
            'Plant resistant tomato varieties.',
            'Remove and destroy infected plants immediately.',
            'Use insect-proof netting in seedling production.'
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Tomato mosaic virus (ToMV) causes mottled light and dark green patterns on leaves, leaf distortion, and reduced fruit quality. It is highly contagious and can spread through handling.',
        'prevention': [
            'Plant resistant varieties.',
            'Disinfect tools and hands before handling plants.',
            'Avoid smoking near tomato plants (tobacco can carry related viruses).',
            'Remove and destroy infected plants.',
            'Use certified virus-free seeds.'
        ]
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial spot in bell peppers is caused by Xanthomonas species. It produces small, water-soaked lesions on leaves that turn brown with yellow halos. Fruit spots are raised and scabby.',
        'prevention': [
            'Use pathogen-free seeds and transplants.',
            'Apply copper-based sprays preventatively.',
            'Rotate crops on a 2-3 year cycle.',
            'Avoid overhead irrigation.',
            'Remove and destroy infected plant debris.'
        ]
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial spot of peach is caused by Xanthomonas arboricola pv. pruni. It causes angular purple-black spots on leaves, twig cankers, and cracked, sunken spots on fruit.',
        'prevention': [
            'Plant resistant varieties.',
            'Apply copper sprays during dormancy.',
            'Avoid planting in wind-exposed sites.',
            'Ensure good air circulation through pruning.',
            'Remove infected twigs and cankers during dormant pruning.'
        ]
    },
    'Strawberry___Leaf_scorch': {
        'description': 'Strawberry leaf scorch is caused by the fungus Diplocarpon earlianum. It produces small, irregular, dark purple spots that enlarge and merge, giving leaves a scorched appearance.',
        'prevention': [
            'Plant resistant strawberry varieties.',
            'Remove and destroy infected leaves.',
            'Ensure proper spacing for air circulation.',
            'Avoid overhead irrigation.',
            'Apply fungicides during the growing season if needed.'
        ]
    },
    'Squash___Powdery_mildew': {
        'description': 'Powdery mildew on squash is caused by Erysiphe cichoracearum or Podosphaera xanthii. It produces white, powdery fungal growth on leaf surfaces, reducing photosynthesis and plant vigor.',
        'prevention': [
            'Plant resistant varieties.',
            'Apply fungicides at first sign of disease.',
            'Ensure good air circulation between plants.',
            'Avoid excessive nitrogen fertilization.',
            'Water at the base of plants, not on foliage.'
        ]
    },

    'Blueberry___healthy': {
        'description': 'The blueberry foliage appears vigorous and healthy with uniform deep green coloration and no fungal leaf spots or rust.',
        'prevention': [
            'Maintain acidic soil conditions (pH 4.5 - 5.2) for optimal micronutrient uptake.',
            'Apply pine bark or peat mulch around root zones to retain soil moisture.',
            'Ensure adequate drip irrigation, avoiding wetting leaves.',
            'Prune mature woody canes periodically to encourage vigorous new shoot growth.'
        ]
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Powdery mildew on cherries is caused by Podosphaera clandestina. It produces distinctive white powdery fungal growth on leaves and shoot tips, causing distortion and curled foliage.',
        'prevention': [
            'Prune inner branches during dormancy to improve sunlight penetration and air circulation.',
            'Apply preventative sulfur or horticultural oil sprays during early shoot development.',
            'Avoid excessive nitrogen fertilizers that stimulate rapid susceptible tender growth.',
            'Remove and destroy heavily infested shoot tips.'
        ]
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'The cherry tree foliage displays healthy leaf structure with vibrant green coloration and no visible mildew, leaf spot, or pest defoliation.',
        'prevention': [
            'Maintain regular orchard sanitation by removing fallen leaves and mummified fruits.',
            'Ensure balanced fertilization and proper seasonal irrigation.',
            'Monitor regularly for early leaf-chewing insects and fungal leaf spot.'
        ]
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Huanglongbing (HLB / Citrus Greening) is a severe bacterial disease spread by Asian citrus psyllids. It causes asymmetrical yellow mottling on leaves, stunted shoots, and lopsided bitter fruit.',
        'prevention': [
            'Inspect regularly for Asian citrus psyllid vectors and apply recommended control treatments.',
            'Only plant certified disease-free nursery stock from certified greenhouses.',
            'Immediately remove and destroy confirmed infected trees to prevent grove-wide spread.',
            'Provide foliar micronutrient applications to maintain tree vigor.'
        ]
    },
    'Peach___healthy': {
        'description': 'The peach foliage shows healthy elongated leaves with smooth margins and normal green pigmentation, free of bacterial spots or leaf curl deformities.',
        'prevention': [
            'Apply dormant copper sprays in late winter to prevent peach leaf curl.',
            'Prune tree canopy in an open vase shape to maximize ventilation and rapid drying.',
            'Maintain weed-free tree basins and consistent irrigation during fruit sizing.'
        ]
    },
    'Pepper,_bell___healthy': {
        'description': 'The bell pepper foliage shows healthy green leaves without bacterial lesions, mosaic virus mottling, or fungal spots.',
        'prevention': [
            'Water at the base using drip irrigation; keep leaves dry.',
            'Use organic mulch around plant bases to prevent soil splashing onto lower leaves.',
            'Rotate pepper and solanaceous crops on a 3-year cycle.',
            'Maintain balanced calcium and magnesium levels in soil.'
        ]
    },
    'Potato___healthy': {
        'description': 'The potato foliage exhibits healthy, lush dark green compound leaves without necrotic target spots or blight lesions.',
        'prevention': [
            'Hill soil properly around developing tubers and stems.',
            'Inspect foliage weekly for early signs of Colorado potato beetle or blight.',
            'Ensure proper plant spacing for canopy air flow.',
            'Rotate fields with legumes, brassicas, or grasses.'
        ]
    },
    'Raspberry___healthy': {
        'description': 'The raspberry plant displays vibrant green serrated leaves with strong canes and no signs of cane blight, rust, or spider mite webbing.',
        'prevention': [
            'Prune out spent floricanes right after fruiting to promote primocane growth and ventilation.',
            'Trellis canes to keep foliage off the ground and improve air flow.',
            'Apply drip irrigation and maintain organic mulch.'
        ]
    },
    'Soybean___healthy': {
        'description': 'The soybean plant exhibits healthy trifoliate leaves with vibrant green coloration and no signs of soybean rust, bacterial blight, or frogeye leaf spot.',
        'prevention': [
            'Rotate soybean with corn or small grains to break pest and disease cycles.',
            'Select certified disease-resistant high-yielding varieties.',
            'Scout fields systematically during flowering and pod development stages.'
        ]
    },
    'Strawberry___healthy': {
        'description': 'The strawberry plant exhibits healthy trifoliate leaves with bright green luster and no signs of leaf scorch, angular leaf spot, or powdery mildew.',
        'prevention': [
            'Plant in raised beds with clean straw mulch to keep leaves and fruits off soil.',
            'Drip irrigate early in the day to minimize leaf wetness duration.',
            'Renovate strawberry beds after harvest and remove old diseased foliage.'
        ]
    },
    'Apple___healthy': {
        'description': 'The apple leaves appear completely healthy with uniform green color, smooth margins, and no signs of scab lesions, rust spots, or powdery mildew.',
        'prevention': [
            'Maintain an open canopy with annual dormant pruning.',
            'Rake and compost or destroy fallen leaves in autumn to eliminate overwintering fungal spores.',
            'Apply dormant horticultural oils to suppress overwintering mites and scale insects.'
        ]
    },
    'Corn_(maize)___healthy': {
        'description': 'The maize/corn leaf shows healthy parallel venation with rich green color and no rust pustules, northern leaf blight lesions, or gray leaf spots.',
        'prevention': [
            'Plant high-performing disease-resistant hybrid seed.',
            'Practice continuous crop rotation with non-host crops like legumes.',
            'Ensure balanced nitrogen, phosphorus, and potassium soil fertility.'
        ]
    },
    'Grape___healthy': {
        'description': 'The grapevine foliage displays healthy lobed green leaves without black rot lesions, downy mildew patches, or leafhopper damage.',
        'prevention': [
            'Perform canopy management (shoot positioning and leaf pulling) around grape clusters.',
            'Apply preventative fungicide sprays according to regional spray schedules.',
            'Disinfect pruning shears regularly when pruning dormant vines.'
        ]
    },
    'Tomato___healthy': {
        'description': 'The tomato foliage displays vibrant, healthy compound leaves with no signs of early blight target spots, bacterial spots, or viral curling.',
        'prevention': [
            'Stake or cage plants to maintain upright growth and prevent soil contact.',
            'Prune lower sucker leaves up to 12 inches from ground level to prevent soil splash infection.',
            'Always water at the base of the plant using drip or soaker hoses.',
            'Apply organic straw or woodchip mulch around root zones.'
        ]
    },
}

# Default information for unknown diseases or healthy plants
_DEFAULT_DISEASED = {
    'description': 'A plant disease has been detected based on visual symptoms observed in the leaf image. The specific condition may affect plant health and yield if left untreated.',
    'prevention': [
        'Consult with a local agricultural extension service for accurate identification.',
        'Remove and destroy heavily infected plant parts.',
        'Ensure proper plant spacing for good air circulation.',
        'Avoid overhead watering to reduce moisture on foliage.',
        'Practice crop rotation to reduce disease pressure.',
        'Consider applying appropriate fungicides or treatments as recommended by experts.'
    ]
}

_DEFAULT_HEALTHY = {
    'description': 'The leaf appears healthy with no visible signs of disease. The plant shows normal coloration and structure.',
    'prevention': [
        'Continue regular watering and balanced fertilization.',
        'Monitor plants regularly for early signs of disease or pest damage.',
        'Maintain good air circulation through proper spacing.',
        'Practice crop rotation each growing season.',
        'Keep the growing area clean and free of debris.',
        'Apply preventive treatments if disease pressure is known in your area.'
    ]
}


def get_disease_info(class_name: str) -> dict:
    """
    Get disease description and prevention tips for a given class name.

    Args:
        class_name: The predicted class name (e.g., 'Tomato___Early_blight').

    Returns:
        Dict with 'description' (str) and 'prevention' (list of str) keys.
    """
    # Check exact match in database
    if class_name in DISEASE_INFO:
        return DISEASE_INFO[class_name]

    # Check if it's a healthy class
    if 'healthy' in class_name.lower():
        return _DEFAULT_HEALTHY

    # Return default for unknown diseases
    return _DEFAULT_DISEASED
