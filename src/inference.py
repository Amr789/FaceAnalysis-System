import torch
import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from src.networks import AgeEstimator

class UTKFacePipeline:
    def __init__(self, age_model_path="models/utk_age_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # A. Face Detector & Matcher
        self.mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=self.device)
        self.matcher = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # B. Age Estimator
        self.age_model = AgeEstimator(pretrained=False) # No need to download ImageNet weights again
        if os.path.exists(age_model_path):
            self.age_model.load_state_dict(torch.load(age_model_path, map_location=self.device))
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

    def compare(self, path1, path2):
        t1_match, t1_age = self.process_image(path1)
        t2_match, t2_age = self.process_image(path2)

        if t1_match is None or t2_match is None:
            return {"Error": "No face detected"}

        with torch.no_grad():
            age1 = self.age_model(t1_age).item()
            age2 = self.age_model(t2_age).item()
            
            emb1 = self.matcher(t1_match)
            emb2 = self.matcher(t2_match)
            
            sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()

        return {
            "age_1": round(age1, 1),
            "age_2": round(age2, 1),
            "similarity": round(sim, 4),
            "match": sim > 0.6
        }