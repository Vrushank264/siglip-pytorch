import torch
from PIL import Image
from torchvision import transforms
from configs.configs import VisionConfig, Qwen3Config
from models.siglip import SigLIP
from transformers import AutoTokenizer


class_names = ["A photo of a house", "A photo of a dog", "A photo of a San Franscisco city", "A photo of London city"]

image_path = "/home/ec2-user/SageMaker/photo.jpg"

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image = Image.open(image_path).convert("RGB")
pixel_values = preprocess(image).unsqueeze(0) 

text_cfg = Qwen3Config()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
inputs = tokenizer(class_names, padding="max_length", max_length=77, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]


vision_cfg = VisionConfig()
model = SigLIP(vision_cfg, text_cfg, embed_dim=768)
ckpt = torch.load("checkpoints/best.pt", map_location="cuda:0")
if ckpt:
    model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

with torch.no_grad():
    img_emb, txt_emb = model(pixel_values, input_ids, attention_mask)
    similarity = (img_emb @ txt_emb.t()).squeeze(0)
    probs = torch.nn.functional.softmax(similarity, dim = 0)
    pred_idx = similarity.argmax().item()
    print(f"Predicted class: {class_names[pred_idx]}")
    print("Similarities:", similarity.tolist())
    print("Probs: ", probs)