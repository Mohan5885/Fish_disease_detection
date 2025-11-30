import argparse
from PIL import Image
import torch
from torchvision import transforms
from model import build_model

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='path to checkpoint .pth')
    p.add_argument('--image', required=True, help='image path')
    p.add_argument('--model_name', default='resnet18', help='timm model name used at training')
    p.add_argument('--topk', type=int, default=1)
    return p.parse_args()

def main():
    args = parse_args()
    ckpt = torch.load(args.model, map_location='cpu')
    classes = ckpt.get('classes', ['class0', 'class1'])
    model = build_model(model_name=args.model_name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    img = load_image(args.image)
    inp = tfms(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(inp)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        topk = probs.topk(args.topk)
        for i, p in zip(topk.indices.tolist(), topk.values.tolist()):
            print(f"Class: {classes[i]}  Prob: {p:.4f}")

if __name__ == '__main__':
    main()
