import os
import argparse
from dataset import FishDataset
from model import build_model
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--model', type=str, default='resnet18')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--output_dir', type=str, default='outputs')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--pretrained', action='store_true', help='use pretrained weights')
    return p.parse_args()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds = []
    trues = []
    for images, labels in tqdm(loader, desc='Train', leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds += outputs.argmax(dim=1).cpu().tolist()
        trues += labels.cpu().tolist()
    acc = accuracy_score(trues, preds) if len(trues) else 0.0
    return (sum(losses)/len(losses)) if losses else 0.0, acc

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    trues = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds += outputs.argmax(dim=1).cpu().tolist()
            trues += labels.cpu().tolist()
    acc = accuracy_score(trues, preds) if len(trues) else 0.0
    return (sum(losses)/len(losses)) if losses else 0.0, acc

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = FishDataset(args.data_dir, split='train', transform=train_tfms)
    val_ds = FishDataset(args.data_dir, split='val', transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(train_ds.classes)
    model = build_model(model_name=args.model, num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} — Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'classes': train_ds.classes
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'best_model.pth'))
    print('Training complete — best val acc:', best_acc)

if __name__ == '__main__':
    main()
