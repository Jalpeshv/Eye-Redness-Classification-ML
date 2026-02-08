import gradio as gr
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import joblib
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from PIL import Image
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
 
# Try to import EfficientNet
try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    print("EfficientNet not available - will skip EfficientNet model")
    EFFICIENTNET_AVAILABLE = False

# Global variables for loaded models
loaded_models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'Normal',
    'Bulbar Conjunctival Redness',
    'Palpebral Conjunctiva Redness',
    'Sub Conjunctival Hemorrhage'
]

# ===========================
# MODEL ARCHITECTURES
# ===========================

class MobileNetV3EyeClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(MobileNetV3EyeClassifier, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class SimpleMobileNetV3(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleMobileNetV3, self).__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        self.backbone.classifier = nn.Linear(960, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# ===========================
# MODEL LOADING FUNCTIONS
# ===========================

def load_random_forest_model():
    """Load Random Forest model and its components"""
    try:
        # Check for model files in current directory
        model_paths = [
            'random_forest_model.joblib',
            'enhanced_random_forest_model.joblib',
            'MODEL/random_forest_model.joblib'
        ]
        
        feature_params_paths = [
            'feature_params.joblib',
            'enhanced_feature_params.joblib', 
            'MODEL/feature_params.joblib'
        ]
        
        model_path = None
        feature_params_path = None
        
        for mp, fp in zip(model_paths, feature_params_paths):
            if os.path.exists(mp) and os.path.exists(fp):
                model_path = mp
                feature_params_path = fp
                break
        
        if model_path and feature_params_path:
            model = joblib.load(model_path)
            feature_params = joblib.load(feature_params_path)
            
            selector = None
            selector_paths = ['feature_selector.joblib', 'MODEL/feature_selector.joblib']
            for sp in selector_paths:
                if os.path.exists(sp):
                    selector = joblib.load(sp)
                    break
            
            return {
                'model': model,
                'feature_params': feature_params,
                'selector': selector,
                'status': 'loaded',
                'model_path': model_path
            }
        else:
            return {'status': 'not_found'}
    except Exception as e:
        print(f"Error loading Random Forest: {e}")
        return {'status': 'error', 'error': str(e)}

def load_efficientnet_model():
    """Load EfficientNet-B3 model"""
    if not EFFICIENTNET_AVAILABLE:
        return {'status': 'not_available'}
        
    try:
        model_paths = [
            'efficientnet_b3_final_model.pth',
            'efficientnet_b3_best_model.pth',
            'MODEL/efficientnet_b3_final_model.pth',
            'MODEL/efficientnet_b3_best_model.pth'
        ]
        
        model_path = None
        for mp in model_paths:
            if os.path.exists(mp):
                model_path = mp
                break
        
        if model_path:
            model = EfficientNet.from_pretrained('efficientnet-b3')
            in_features = model._fc.in_features
            model._fc = nn.Linear(in_features, 4)
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            
            return {
                'model': model,
                'status': 'loaded',
                'model_path': model_path
            }
        else:
            return {'status': 'not_found'}
    except Exception as e:
        print(f"Error loading EfficientNet: {e}")
        return {'status': 'error', 'error': str(e)}

def load_mobilenet_model():
    """Load MobileNet model with enhanced compatibility"""
    try:
        model_paths = [
            'mobilenet_final_model.pth',
            'mobilenet_best_model.pth',
            'MODEL/mobilenet_final_model.pth',
            'MODEL/mobilenet_best_model.pth'
        ]
        
        model_path = None
        for mp in model_paths:
            if os.path.exists(mp):
                model_path = mp
                print(f"Found MobileNet model at: {model_path}")
                break
        
        if not model_path:
            print("No MobileNet model files found")
            return {'status': 'not_found'}
        
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Try different model architectures
        architectures_to_try = [
            ("MobileNetV3EyeClassifier", MobileNetV3EyeClassifier),
            ("SimpleMobileNetV3", SimpleMobileNetV3),
        ]
        
        for arch_name, arch_class in architectures_to_try:
            try:
                print(f"Trying architecture: {arch_name}")
                model = arch_class(num_classes=4)
                
                # Try loading state dict
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                model.load_state_dict(state_dict, strict=False)
                model.to(device)
                model.eval()
                
                print(f"✅ Successfully loaded MobileNet with {arch_name}")
                return {
                    'model': model,
                    'status': 'loaded',
                    'model_path': model_path,
                    'architecture': arch_name
                }
                
            except Exception as e:
                print(f"Failed to load with {arch_name}: {str(e)}")
                continue
        
        return {'status': 'error', 'error': 'All architectures failed'}
            
    except Exception as e:
        print(f"Error in load_mobilenet_model: {str(e)}")
        return {'status': 'error', 'error': str(e)}

# ===========================
# FEATURE EXTRACTION FOR RANDOM FOREST
# ===========================

def extract_features(image_array, target_size=(128, 128)):
    """Extract features from image array for Random Forest model"""
    try:
        # Convert to PIL Image if needed
        if isinstance(image_array, np.ndarray):
            if image_array.dtype == np.float32 or image_array.dtype == np.float64:
                image_array = (image_array * 255).astype(np.uint8)
            img = Image.fromarray(image_array)
        else:
            img = image_array
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize and convert to numpy array
        img = img.resize(target_size)
        img = np.array(img)
        
        features = []
        
        # Color histograms (RGB channels)
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
            
        # Texture features using Haralick texture
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # GLCM features
        glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        
        features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
        
        # Edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_count = np.count_nonzero(edges)
        features.append(edge_count / (target_size[0] * target_size[1]))
        
        # Statistical features
        for i in range(3):
            channel = img[:,:,i]
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.min(channel),
                np.max(channel)
            ])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# ===========================
# DEMO/FALLBACK PREDICTIONS
# ===========================

def generate_demo_prediction():
    """Generate a demo prediction when models are not available"""
    # Simulate realistic predictions
    predictions = np.random.dirichlet([2, 1, 1, 1])  # Slightly favor "Normal"
    result = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return result

# ===========================
# PREDICTION FUNCTIONS
# ===========================

def predict_with_random_forest(image, model_data):
    """Predict using Random Forest model"""
    try:
        target_size = model_data['feature_params'].get('target_size', (128, 128))
        features = extract_features(image, target_size=target_size)
        
        if features is not None:
            if model_data['selector'] is not None:
                features = model_data['selector'].transform([features])
            else:
                features = [features]
            
            probabilities = model_data['model'].predict_proba(features)[0]
            result = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
            
            return result
        else:
            return {class_name: 0.0 for class_name in class_names}
            
    except Exception as e:
        print(f"Random Forest prediction error: {e}")
        return {"Error": f"Prediction failed: {str(e)}"}

def predict_with_pytorch_model(image, model_data, model_type):
    """Predict using PyTorch models (EfficientNet or MobileNet)"""
    try:
        if model_type == "EfficientNet-B3":
            transform = transforms.Compose([
                transforms.ToPILImage() if isinstance(image, np.ndarray) else transforms.Lambda(lambda x: x),
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:  # MobileNet
            transform = transforms.Compose([
                transforms.ToPILImage() if isinstance(image, np.ndarray) else transforms.Lambda(lambda x: x),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Handle different image formats
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model_data['model'](input_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        result = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        return result
        
    except Exception as e:
        print(f"PyTorch model prediction error: {e}")
        return {"Error": f"Prediction failed: {str(e)}"}

# ===========================
# MAIN GRADIO INTERFACE
# ===========================

def initialize_models():
    """Initialize all available models"""
    global loaded_models
    
    print("🔄 Loading models...")
    
    # Load Random Forest
    rf_data = load_random_forest_model()
    if rf_data['status'] == 'loaded':
        loaded_models['Random Forest'] = rf_data
        print("✅ Random Forest model loaded successfully")
    else:
        print("⚠️ Random Forest model not found - will use demo mode")
    
    # Load EfficientNet-B3
    if EFFICIENTNET_AVAILABLE:
        efficientnet_data = load_efficientnet_model()
        if efficientnet_data['status'] == 'loaded':
            loaded_models['EfficientNet-B3'] = efficientnet_data
            print("✅ EfficientNet-B3 model loaded successfully")
        else:
            print("⚠️ EfficientNet-B3 model not found - will use demo mode")
    else:
        print("⚠️ EfficientNet not available - skipping")
    
    # Load MobileNet
    mobilenet_data = load_mobilenet_model()
    if mobilenet_data['status'] == 'loaded':
        loaded_models['MobileNet-V3'] = mobilenet_data
        print("✅ MobileNet-V3 model loaded successfully")
    else:
        print("⚠️ MobileNet-V3 model not found - will use demo mode")
    
    # Always include demo models for demonstration
    available_models = ['Random Forest', 'EfficientNet-B3', 'MobileNet-V3']
    
    return available_models

def predict_image(image, model_choice):
    """Main prediction function that routes to the appropriate model"""
    
    if image is None:
        return {"Error": "Please upload an image"}
    
    try:
        # Check if real model is available
        if model_choice in loaded_models:
            if model_choice == 'Random Forest':
                return predict_with_random_forest(image, loaded_models[model_choice])
            elif model_choice == 'EfficientNet-B3':
                return predict_with_pytorch_model(image, loaded_models[model_choice], "EfficientNet-B3")
            elif model_choice == 'MobileNet-V3':
                return predict_with_pytorch_model(image, loaded_models[model_choice], "MobileNet-V3")
        else:
            # Use demo mode
            print(f"Using demo mode for {model_choice}")
            demo_result = generate_demo_prediction()
            demo_result["Note"] = f"Demo mode - {model_choice} model not available"
            return demo_result
            
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"Error": f"Prediction error: {str(e)}"}

def get_model_info(model_choice):
    """Return information about the selected model"""
    
    model_info = {
        'Random Forest': {
            'description': 'Traditional machine learning model using hand-crafted features including color histograms, texture features (GLCM), and edge features.',
            'features': 'Color histograms (RGB), GLCM texture features, edge density, statistical features',
            'training_time': 'Very fast (~10 seconds)',
            'accuracy': 'Good baseline performance, interpretable features',
            'advantages': 'Fast training, interpretable, works well with limited data'
        },
        'EfficientNet-B3': {
            'description': 'State-of-the-art deep learning model optimized for efficiency and accuracy using compound scaling.',
            'features': 'Learned features through efficient convolutional neural networks',
            'training_time': 'Moderate (~1-2 hours)',
            'accuracy': 'High accuracy with excellent feature learning capabilities',
            'advantages': 'Best accuracy, efficient architecture, transfer learning'
        },
        'MobileNet-V3': {
            'description': 'Lightweight deep learning model optimized for mobile deployment using depthwise separable convolutions.',
            'features': 'Learned features with mobile-optimized architecture',
            'training_time': 'Fast (~30-60 minutes)',
            'accuracy': 'Good balance of accuracy and computational efficiency',
            'advantages': 'Lightweight, fast inference, mobile-friendly'
        }
    }
    
    if model_choice in model_info:
        info = model_info[model_choice]
        is_loaded = model_choice in loaded_models
        status = '✅ Loaded' if is_loaded else '⚠️ Demo Mode'
        
        # Add architecture info for MobileNet if available
        extra_info = ""
        if model_choice == 'MobileNet-V3' and is_loaded:
            arch = loaded_models[model_choice].get('architecture', 'unknown')
            extra_info = f"\n\n🏗️ **Architecture:** {arch}"
        
        return f"""## **{model_choice} Information**

📋 **Description:** {info['description']}

🔧 **Features:** {info['features']}

⏱️ **Training Time:** {info['training_time']}

📊 **Performance:** {info['accuracy']}

✨ **Advantages:** {info['advantages']}{extra_info}

🔄 **Status:** {status}"""
    else:
        return "No information available for this model."

def create_interface():
    """Create the main Gradio interface"""
    
    # Initialize models
    available_models = initialize_models()
    
    print(f"📋 Available models: {', '.join(available_models)}")
    
    with gr.Blocks(
        title="Multi-Model Eye Redness Classification",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .model-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #007bff;
        }
        """
    ) as iface:
        
        gr.Markdown("""
        # 🏥 Multi-Model Eye Redness Classification System
        
        **Upload an eye image and select a model to classify the type of redness.**
        
        This system compares different machine learning approaches: **Random Forest**, **EfficientNet-B3**, and **MobileNet-V3**.
        
        🎯 **Classes**: Normal, Bulbar Conjunctival Redness, Palpebral Conjunctiva Redness, Sub Conjunctival Hemorrhage
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0] if available_models else None,
                    label="🤖 Select Model",
                    info="Choose which model to use for classification"
                )
                
                image_input = gr.Image(
                    label="📷 Upload Eye Image",
                    type="numpy",
                    height=300
                )
                
                predict_btn = gr.Button(
                    "🔍 Classify Image",
                    variant="primary",
                    size="lg"
                )
        
            with gr.Column(scale=1):
                model_info_display = gr.Markdown(
                    get_model_info(available_models[0] if available_models else ""),
                    elem_classes=["model-info"]
                )
        
        with gr.Row():
            results_output = gr.Label(
                num_top_classes=4,
                label="📊 Classification Results",
                show_label=True
            )
        
        with gr.Row():
            gr.Markdown("""
            ### 📝 Instructions:
            
            1. **Select a Model**: Choose from Random Forest, EfficientNet-B3, or MobileNet-V3
            2. **Upload Image**: Click on the image area to upload a clear eye image
            3. **Classify**: Click "Classify Image" to get predictions
            4. **Compare Models**: Try the same image with different models to compare results
            
            ### 💡 Tips for Best Results:
            - Use clear, well-lit images with the eye clearly visible
            - Ensure the eye takes up a significant portion of the image
            - Avoid blurry or low-resolution images
            - Different models may perform better on different types of images
            
            ### ⚠️ Disclaimer:
            This is a research tool for educational purposes. **Always consult healthcare professionals for medical diagnosis.**
            """)
        
        # Event handlers
        model_dropdown.change(
            fn=get_model_info,
            inputs=[model_dropdown],
            outputs=[model_info_display]
        )
        
        predict_btn.click(
            fn=predict_image,
            inputs=[image_input, model_dropdown],
            outputs=[results_output]
        )
        
        # Allow prediction on image upload as well
        image_input.change(
            fn=predict_image,
            inputs=[image_input, model_dropdown],
            outputs=[results_output]
        )
    
    return iface

# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == "__main__":
    print("🚀 Starting Multi-Model Eye Redness Classification System...")
    print(f"🔧 Using device: {device}")
    
    interface = create_interface()
    
    if interface is not None:
        print("✅ Interface created successfully!")
        print("🌐 Launching application...")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Set to True if you want a public link
            show_error=True,
            show_tips=True
        )
    else:
        print("❌ Failed to create interface.")
