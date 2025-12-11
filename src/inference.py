import torch
import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from src.networks import AgeEstimator

class UTKFacePipeline:
    def __init__(self, age_model_path="models/utk_age_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing pipeline on: {self.device}")
        
        # A. Face Detector & Matcher
        self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=self.device)
        self.matcher = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # B. Age Estimator (Updated to EfficientNet)
        self.age_model = AgeEstimator(pretrained=False) 
        
        if os.path.exists(age_model_path):
            try:
                state_dict = torch.load(age_model_path, map_location=self.device)
                self.age_model.load_state_dict(state_dict)
                print("Loaded EfficientNet age model.")
            except RuntimeError:
                print("ERROR: Weight mismatch! You are trying to load ResNet weights into EfficientNet.")
                print("Please retrain the model before running inference.")
        else:
            print("Warning: Age model weights not found!")
            
        self.age_model.to(self.device).eval()

    def process_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            return None, None

        # 1. Get Face (160x160) for Matching
        face_tensor_match = self.mtcnn(img)
        if face_tensor_match is None: return None, None

        # 2. Resize (224x224) for Age
        face_tensor_age = torch.nn.functional.interpolate(
            face_tensor_match.unsqueeze(0), size=(224, 224), mode='bilinear'
        )

        return face_tensor_match.unsqueeze(0).to(self.device), face_tensor_age.to(self.device)

    def predict_age_with_tta(self, face_tensor):
        """
        Runs Test Time Augmentation (TTA).
        Predicts on original image AND horizontally flipped image, then averages.
        """
        # 1. Original Prediction
        pred_orig = self.age_model(face_tensor).item()
        
        # 2. Flipped Prediction (Horizontal Flip on width dim)
        # Tensor shape is [Batch, Channel, Height, Width], so we flip dim 3
        face_flipped = torch.flip(face_tensor, [3])
        pred_flip = self.age_model(face_flipped).item()
        
        # 3. Average
        return (pred_orig + pred_flip) / 2.0

    def compare(self, path1, path2):
        t1_match, t1_age = self.process_image(path1)
        t2_match, t2_age = self.process_image(path2)

        if t1_match is None or t2_match is None:
            return {"Error": "No face detected"}

        with torch.no_grad():
            # Use TTA here
            age1 = self.predict_age_with_tta(t1_age)
            age2 = self.predict_age_with_tta(t2_age)
            
            emb1 = self.matcher(t1_match)
            emb2 = self.matcher(t2_match)
            
            sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()

        return {
            "age_1": round(age1, 1),
            "age_2": round(age2, 1),
            "similarity": round(sim, 4),
            "match": sim > 0.6
        }
