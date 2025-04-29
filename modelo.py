import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizando dispositivo {device}")

# Dataset personalizado para carregar espectrogramas
class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels):
        self.spectrograms = spectrograms
        self.labels = labels
        
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        spectrogram = self.spectrograms[idx]
        label = self.labels[idx]
        
        # Adiciona dimensão de canal (para CNN) e converte para tensor do torch
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        return spectrogram, label

# Modelo CNN
class AudioCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(AudioCNN, self).__init__()
        
        # Camadas convolucionais
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Calculando o tamanho após convoluções e pooling
        # Para espectrogramas Mel típicos de tamanho (128, ~130)
        self.fc1_input_features = self._get_fc_input_size()
        
        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(self.fc1_input_features, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def _get_fc_input_size(self):
        # Estima o tamanho após convoluções - será substituído no forward
        # Suposição padrão para tamanho do espectrograma mel (1, 128, 130)
        return 64 * 16 * 16
        
    def forward(self, x):
        # Camadas convolucionais
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Atualiza o tamanho de entrada de FC com base nas dimensões reais do tensor
        batch_size = x.size(0)
        self.fc1_input_features = x.size(1) * x.size(2) * x.size(3)
        
        # Achata para camadas totalmente conectadas
        x = x.view(batch_size, -1)
        
        # Se primeira execução, ajusta a camada FC1
        if not hasattr(self, 'fc1') or self.fc1.in_features != self.fc1_input_features:
            self.fc1 = nn.Linear(self.fc1_input_features, 128).to(x.device)
            
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_data(base_folder, classes=['background', 'edjo', 'sao_mag_cenas']):
    """
    Load spectrograms and labels from the processed data folders
    """
    all_spectrograms = []
    all_labels = []
    
    for idx, class_name in enumerate(classes):
        spec_folder = os.path.join(base_folder, f"espectrogramas_{class_name}")
        
        if not os.path.exists(spec_folder):
            print(f"Warning: Folder {spec_folder} not found!")
            continue
        
        file_list = [f for f in os.listdir(spec_folder) if f.endswith('_melspec.npy')]
        
        for file_name in file_list:
            spec_path = os.path.join(spec_folder, file_name)
            try:
                # Load the spectrogram
                spectrogram = np.load(spec_path)
                
                # Add to dataset
                all_spectrograms.append(spectrogram)
                all_labels.append(idx)  # Use class index as label
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    return np.array(all_spectrograms), np.array(all_labels)

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=20):
    """
    Train the CNN model
    """
    train_losses = []
    valid_losses = []
    valid_accs = []
    
    best_valid_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for spectrograms, labels in train_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * spectrograms.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for spectrograms, labels in valid_loader:
                spectrograms, labels = spectrograms.to(device), labels.to(device)
                
                outputs = model(spectrograms)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * spectrograms.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_valid_loss = running_loss / len(valid_loader.dataset)
        epoch_valid_acc = correct / total
        
        valid_losses.append(epoch_valid_loss)
        valid_accs.append(epoch_valid_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Valid Loss: {epoch_valid_loss:.4f}, "
              f"Valid Acc: {epoch_valid_acc:.4f}")
        
        # Save best model
        if epoch_valid_acc > best_valid_acc:
            best_valid_acc = epoch_valid_acc
            torch.save(model.state_dict(), 'best_audio_cnn.pth')
    
    return train_losses, valid_losses, valid_accs

def evaluate_model(model, test_loader, class_names):
    """
    Evaluate the model on test data and display results
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            outputs = model(spectrograms)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return all_preds, all_labels

def main():
    # Parameters
    base_folder = './audio_data'  # Change this to your actual data folder
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    class_names = ['background', 'edjo', 'sao_mag_cenas']
    
    # Load data
    print("Loading spectrograms...")
    spectrograms, labels = load_data(base_folder, class_names)
    
    if len(spectrograms) == 0:
        print("No data found! Please check the paths and preprocess the data first.")
        return
    
    print(f"Loaded {len(spectrograms)} spectrograms with shape {spectrograms[0].shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        spectrograms, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    # Create datasets
    train_dataset = SpectrogramDataset(X_train, y_train)
    valid_dataset = SpectrogramDataset(X_val, y_val)
    test_dataset = SpectrogramDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = AudioCNN(num_classes=len(class_names)).to(device)
    print(model)
    
    # Sample forward pass to check dimensions
    sample_input = next(iter(train_loader))[0][:1].to(device)
    print(f"Sample input shape: {sample_input.shape}")
    with torch.no_grad():
        sample_output = model(sample_input)
    print(f"Sample output shape: {sample_output.shape}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    print("\nTraining model...")
    train_losses, valid_losses, valid_accs = train_model(
        model, train_loader, valid_loader, criterion, optimizer, num_epochs
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(valid_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Load best model
    model.load_state_dict(torch.load('best_audio_cnn.pth'))
    
    # Evaluate model
    print("\nEvaluating model...")
    all_preds, all_labels = evaluate_model(model, test_loader, class_names)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()