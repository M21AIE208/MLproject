#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.optim as optim
from sklearn.decomposition import PCA

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize features
scaler = StandardScaler()


# In[2]:


# Define custom dataset
class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = scaler.fit_transform(X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# In[3]:


# Define MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 5)
        self.fc2 = nn.Linear(5, 7)
        self.fc3 = nn.Linear(7, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, optimizer, and loss function
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# In[8]:


model


# In[4]:


# KFold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
writer = SummaryWriter()  # Initialize Tensorboard writer
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_dataset = IrisDataset(X_train, y_train)
    test_dataset = IrisDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Training
    model.train()
    for epoch in range(10):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        writer.add_scalar('Loss/train', loss.item(), epoch)

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(targets.numpy())
            y_pred.extend(predicted.numpy())

    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    writer.add_scalar('Precision', precision, fold)
    writer.add_scalar('Recall', recall, fold)
    writer.add_scalar('Accuracy', accuracy, fold)




# In[5]:


# Generating PCA and
pca = PCA(n_components=2,
         random_state = 123,
         svd_solver = 'auto'
         )
X_pca = pca.fit_transform(X)
## TensorFlow Variable from data
tf_data = torch.Tensor(X_pca)
writer.add_embedding(tf_data,metadata=y)

writer.close()


# In[6]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[9]:


get_ipython().run_line_magic('tensorboard', '--logdir runs')


# In[7]:




