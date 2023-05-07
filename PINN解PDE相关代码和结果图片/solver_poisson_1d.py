from net import Net
import os
import matplotlib.pyplot as plt
import torch
from torch.autograd import grad


def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]


def PDE(u, x):
    return d(d(u, x), x) + 0.49 * torch.sin(0.7 * x) + 2.25 * torch.cos(1.5 * x)


def Ground_true(x):
    return torch.sin(0.7 * x) + torch.cos(1.5 * x) - 0.1 * x


def train():
    x_left, x_right = -10, 10
    lr = 0.001
    n_pred = 100
    n_f = 200
    epochs = 8000

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    PINN = Net([1, 50, 50, 50, 1]).to(device)

    optimizer = torch.optim.Adam(PINN.parameters(), lr)
    criterion = torch.nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        # inside
        x_f = ((x_left + x_right) / 2 + (x_right - x_left) *
               (torch.rand(size=(n_f, 1), dtype=torch.float, device=device) - 0.5)
               ).requires_grad_(True)
        u_f = PINN(x_f)
        PDE_ = PDE(u_f, x_f)
        mse_PDE = criterion(PDE_, torch.zeros_like(PDE_))

        # boundary
        x_b = torch.tensor([[-10.], [10.]]).requires_grad_(True).to(device)
        u_b = PINN(x_b)
        true_b = Ground_true(x_b)
        mse_BC = criterion(u_b, true_b)

        # predict
        x_pred = ((x_left + x_right) / 2 + (x_right - x_left) *
                  (torch.rand(size=(n_pred, 1), dtype=torch.float, device=device) - 0.5)
                  ).requires_grad_(True)
        u_f = PINN(x_pred)
        true_f = Ground_true(x_pred)
        mse_pred = criterion(u_f, true_f)

        loss = 1 * mse_PDE + 1 * mse_BC
        loss_history.append([mse_PDE.item(), mse_BC, mse_pred])

        if epoch % 100 == 0:
            print(
                'epoch:{:05d}, EoM: {:.08e}, BC: {:.08e}, loss: {:.08e}'.format(
                    epoch, mse_PDE.item(), mse_BC.item(), loss.item()
                )
            )

        loss.backward()
        optimizer.step()

        xx = torch.linspace(-10, 10, 10000).reshape((-1, 1)).to(device)

        if (epoch + 1) % 1000 == 0:
            yy = PINN(xx)
            zz = Ground_true(xx)
            xx = xx.reshape((-1)).data.detach().cpu().numpy()
            yy = yy.reshape((-1)).data.detach().cpu().numpy()
            zz = zz.reshape((-1)).data.detach().cpu().numpy()
            plt.cla()

            plt.plot(xx, yy, label='PINN')
            plt.plot(xx, zz, label='True', color='r')
            plt.ylim(-3, 3)
            plt.legend()

            plt.title('PINN(epoch{}))'.format(epoch + 1))
            plt.savefig('./result_plot/poisson1d_{}.png'.format(epoch + 1), bbox_inches='tight', format='png')
            plt.show()

    plt.plot(loss_history)
    plt.legend(('PDE loss', 'BC loss', 'Pred loss'), loc='best')
    plt.yscale('log')
    plt.savefig('./result_plot/poisson1d_loss.png', bbox_inches='tight', format='png')
    plt.show()


if __name__ == '__main__':
    train()
