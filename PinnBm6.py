import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class BrownianPINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BrownianPINN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h1 = torch.sin(self.fc1(x))
        h2 = torch.sin(self.fc2(h1))
        h3 = torch.sin(self.fc3(h2))
        y = self.fc4(h3)
        return y

def train(model, x_train, y_train, loss_fn, optimizer, num_epochs):
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    for epoch in range(num_epochs):
        y_pred = model(x_train)

        loss = loss_fn(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 定义训练数据、损失函数、优化器和训练轮数
t_train = np.linspace(0, 1, 100)[:, None]
x_train = np.random.normal(size=(100, 1))
y_train = np.random.normal(size=(100, 1))

model = BrownianPINN(input_size=2, hidden_size=20, output_size=1)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 1000

# 训练模型
train(model, np.hstack([t_train, x_train]), y_train, loss_fn, optimizer, num_epochs)

# evaluate the model
x_test = np.linspace(0, 1, 100)[:, None]
y_test = np.sin(2 * np.pi * x_test)

with torch.no_grad():
    u_pred = model(torch.tensor(np.hstack([x_test, np.zeros_like(x_test)]), dtype=torch.float32))
    loss = F.mse_loss(u_pred, torch.tensor(y_test, dtype=torch.float32))

print(f"Test loss: {loss:.8f}")

# plot the results
fig, ax = plt.subplots()
ax.plot(x_test, y_test, 'b-', label='Exact')
ax.plot(x_test, u_pred.numpy(), 'r--', label='Predicted')
ax.legend(loc='lower left')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()