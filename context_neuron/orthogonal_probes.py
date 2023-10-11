import torch
import torch.nn as nn
import torch.optim as optim

# Define the logistic regression probe
class Probe(nn.Module):
    def __init__(self, input_dim):
        super(Probe, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Assuming features and labels are prepared
features = torch.tensor(...) # Shape [N, d]
labels1 = torch.tensor(...) # Shape [N]
labels2 = torch.tensor(...) # Shape [N]

# Create the first probe and train it
probe1 = Probe(input_dim=features.size(1))
optimizer1 = optim.SGD(probe1.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    outputs = probe1(features)
    loss = criterion(outputs, labels1)
    loss.backward()
    optimizer1.step()
    optimizer1.zero_grad()

# Create the second probe
probe2 = Probe(input_dim=features.size(1))
optimizer2 = optim.SGD(probe2.parameters(), lr=0.01)

# Orthogonal loss function
def orthogonal_loss(w1, w2):
    cos_similarity = torch.dot(w1, w2) / (torch.norm(w1) * torch.norm(w2))
    return cos_similarity ** 2

# Training loop for the second probe
for epoch in range(epochs):
    outputs = probe2(features)
    loss = criterion(outputs, labels2)
    
    # Add orthogonal loss to the first probe's weights
    w1 = probe1.linear.weight.flatten()
    w2 = probe2.linear.weight.flatten()
    loss += orthogonal_loss(w1, w2)
    
    loss.backward()
    optimizer2.step()
    optimizer2.zero_grad()