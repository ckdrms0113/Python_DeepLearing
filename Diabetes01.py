import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 하이퍼파라미터
class Config:
    # 데이터
    file_path = '/content/diabetes.csv'
    test_size = 0.2
    random_state = 42
    
    # 모델 
    conv1_kernel_size = (2, 1)
    conv2_kernel_size = (2, 1)
    fc1_out_features = 256
    
    # 학습
    batch_size = 32
    learning_rate = 0.005
    num_epochs = 50

# 데이터 로드
df = pd.read_csv(Config.file_path)

# 특성,타겟 변수 분리
X = df.drop(columns='Outcome').values
y = df['Outcome'].values

# 데이터분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.test_size, random_state=Config.random_state)

# 정규화 (스케일링)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 텐서 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# CNN을 2D변환
X_train = X_train.unsqueeze(1).unsqueeze(3)
X_test = X_test.unsqueeze(1).unsqueeze(3)

# 커스텀 Dataset 클래스 정의
class DiabetesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# trans, test dataset
train_dataset = DiabetesDataset(X_train, y_train)
test_dataset = DiabetesDataset(X_test, y_test)

# DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=Config.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=Config.batch_size, shuffle=False)

# CNN model
class DiabetesCNN(nn.Module):
    def __init__(self):
        super(DiabetesCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=Config.conv1_kernel_size)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=Config.conv2_kernel_size)
        self.fc1 = nn.Linear(64 * 6, Config.fc1_out_features)
        self.fc2 = nn.Linear(Config.fc1_out_features, 1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# 모델 초기화
model = DiabetesCNN()

# 손실 함수와 옵티마이저 정의
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)

# 모델 학습
train_losses = []
for epoch in range(Config.num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{Config.num_epochs}], Loss: {epoch_loss:.4f}')

# Loss 그래프 출력
plt.figure(figsize=(10, 5))
plt.plot(range(1, Config.num_epochs+1), train_losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 모델 평가
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
