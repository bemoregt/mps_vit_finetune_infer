import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image

# MPS 디바이스 사용 가능 여부 확인
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 이미지 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 모델 입력 크기에 맞춰 이미지 크기 조정
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 사용자 정의 데이터셋 로드
train_dataset = torchvision.datasets.ImageFolder(root='/Users/jup1/Downloads/hymenoptera_data/train', transform=transform)

# DataLoader 인스턴스 생성
data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 사전학습된 모델 로드
model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

# 분류기 레이어 교체
num_classes = len(train_dataset.classes)

# `model.heads[0].in_features`를 사용하여 새로운 분류기 생성
model.heads = torch.nn.Linear(model.heads[0].in_features, num_classes)

# 모델을 MPS 디바이스로 이동
model = model.to(device)

# 손실 함수와 옵티마이저 정의
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 전이학습 진행
num_epochs = 25
for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0
    
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.float() / len(train_dataset)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

# 학습된 모델 저장
torch.save(model.state_dict(), 'trained_model_vit_b16.pth')

# 저장된 모델 로드
loaded_model = models.vit_b_16(weights=None)
loaded_model.heads = torch.nn.Linear(loaded_model.heads[0].in_features, num_classes)
loaded_model.load_state_dict(torch.load('trained_model_vit_b16.pth'))
loaded_model = loaded_model.to(device)
loaded_model.eval()

# 이미지 추론 함수
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = loaded_model(image)
        _, preds = torch.max(outputs, 1)
    
    predicted_class = train_dataset.classes[preds[0]]
    return predicted_class

# 이미지 경로 설정 및 추론
image_path = '/Users/jup1/Downloads/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
predicted_class = predict_image(image_path)
print(f'Predicted class: {predicted_class}')