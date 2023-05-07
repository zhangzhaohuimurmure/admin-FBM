import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 生成随机漫步数据
N = 1000
T = 1.0
dt = T/N
W = np.zeros((N+1,))
for i in range(1, N+1):
    dW = np.sqrt(dt) * np.random.normal()
    W[i] = W[i-1] + dW

# 定义模型类
class BrownianMotionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BrownianMotionModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# 定义训练函数
def train_model(model, criterion, optimizer, x_train, y_train, num_epochs):
    for epoch in range(num_epochs):
        hidden = torch.zeros(1, 1, model.hidden_size)
        loss = 0
        for i in range(len(x_train)):
            x = torch.Tensor(x_train[i]).unsqueeze(0).unsqueeze(2)
            y_true = torch.Tensor(y_train[i]).unsqueeze(0).unsqueeze(2)
            y_pred, hidden = model(x, hidden)
            loss += criterion(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 训练模型
input_size = 1
hidden_size = 10
output_size = 1
lr = 0.01
num_epochs = 1000

x_train = W[:-1].reshape(-1, 1)
y_train = W[1:].reshape(-1, 1)

model = BrownianMotionModel(input_size, hidden_size, output_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_model(model, criterion, optimizer, x_train, y_train, num_epochs)

# 生成新的随机漫步数据并可视化
x_test = x_train[-1].reshape(-1, 1)
y_test = []
hidden = torch.zeros(1, 1, model.hidden_size)
t = np.linspace(0, T, N+1)

for i in range(N - len(x_train) + 1):
    x = torch.Tensor(x_test[i]).unsqueeze(0).unsqueeze(2)
    y_pred, hidden = model(x, hidden)
    y_test.append(y_pred.detach().numpy()[0][0])

t_test = np.linspace(T, 2*T, N - len(x_train) + 1)
y_test = np.array(y_test)
W_test = np.concatenate([W, y_test.cumsum()])

plt.plot(np.concatenate([t, t_test]), W_test)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Brownian Motion')
plt.show()





# # 生成新的随机漫步数据并可视化
# x_test = x_train[-1].reshape(-1, 1)
# y_test = []
# hidden = torch.zeros(1, 1, model.hidden_size)
# t = np.linspace(0, T, N+1)
#
# for i in range(N - len(x_train) + 1):
#     x = torch.Tensor(x_test[i]).unsqueeze(0).unsqueeze(2)
#     y_pred, hidden = model(x, hidden)
#     y_test.append(y_pred.detach().numpy()[0][0])
#
# y_test = np.array(y_test)
# t_test = np.linspace(T, 2*T, N)
# W_test = np.concatenate([W, y_test.cumsum()])
#
# plt.plot(np.concatenate([t, t_test]), W_test)
# plt.xlabel('Time')
# plt.ylabel('Position')
# plt.title('Brownian Motion')
# plt.show()




"""
# 生成新的随机漫步数据并可视化
x_test = x_train[-1].reshape(-1, 1)
y_test = []
hidden = torch.zeros(1, 1, model.hidden_size)

print(x_test.shape)

for i in range(N):
    x = torch.Tensor(x_test[i]).unsqueeze(0).unsqueeze(2)
    y_pred, hidden = model(x, hidden)
    y_test.append(y_pred.detach().numpy()[0][0])

y_test = np.array(y_test)
t_test = np.linspace(T, 2*T, N)
W_test = np.concatenate([W, y_test.cumsum()])

plt.plot(np.concatenate([t, t_test]), W_test)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Brownian Motion')
plt.show() """
