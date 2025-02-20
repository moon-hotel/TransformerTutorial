import torch
import torch.nn as nn
# import matplotlib.pyplot as plt


class CustomSchedule(nn.Module):
    def __init__(self, d_model, warmup_steps=4000, optimizer=None):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.int32)
        self.warmup_steps = warmup_steps
        self.steps = 1.
        self.optimizer = optimizer

    def step(self):
        arg1 = self.steps ** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1.
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr


# if __name__ == '__main__':
#     for d in [[256, 4000], [512, 4000], [512, 8000]]:
#         lr_list = []
#         lr = CustomSchedule(d[0], warmup_steps=d[1])
#         for i in range(20000):
#             lr_list.append(lr())
#         plt.plot(lr_list, label=f"d_model = {d[0]}, warm_up = {d[1]}")
#     plt.legend()
#     plt.xlabel("Steps")
#     plt.ylabel("Learning rate")
#     plt.show()
