import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from syft.frameworks.torch.crypto import tanaka

# Define a simple DNN model
class MedicalImageClassifier(nn.Module):
    def _init_(self):
        super(MedicalImageClassifier, self)._init_()
        self.fc1 = nn.Linear(784, 128)  # Input: 28x28 image flattened
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)    # Output: Binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

# Preprocess and encrypt data
def preprocess_and_encrypt(image, scheme):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    image_tensor = transform(image).view(-1, 784)  # Flatten the image
    encrypted_image = scheme.encrypt(image_tensor)
    return encrypted_image

# Decrypt results
def decrypt_and_interpret(encrypted_output, scheme):
    decrypted_output = scheme.decrypt(encrypted_output)
    return decrypted_output.argmax(dim=1)  # Return the predicted class

# Load model and define encryption scheme
model = MedicalImageClassifier()
encryption_scheme = tanaka.TanakaScheme()  # Using TANAKA scheme from PySyft

# Example of a single image
image = torch.rand((28, 28))  # Replace with an actual medical image
encrypted_image = preprocess_and_encrypt(image, encryption_scheme)

# Perform encrypted inference
encrypted_output = model(encrypted_image)
predicted_class = decrypt_and_interpret(encrypted_output, encryption_scheme)

print(f"Predicted Class: {predicted_class.item()}")
