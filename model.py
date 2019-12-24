import torch

from dataset import TransDataset
from torch.utils.data import DataLoader


dataset = TransDataset(0)
loader = DataLoader(dataset,
	batch_size=1000,
	shuffle=True,
      	num_workers=1)


model = torch.nn.Sequential(
    torch.nn.Linear(2, 100),
    torch.nn.ReLU(), 
    torch.nn.Linear(100, 1)
)

model.to("cuda:3")
loss_fn = torch.nn.L1Loss(reduction='sum')
max_epoch = 100

trainer = torch.optim.Adam(model.parameters(), lr=0.0003)
for epoch in range(max_epoch):
    loss_total = 0
    for iter_id, batch in enumerate(loader):
        x, y = batch
        x,y  = x.float().to("cuda:3"), y.float().to("cuda:3")

        trainer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
        loss_total += loss
        loss.backward()
        trainer.step()
        if iter_id % 100 == 0:
            print("epoch{}: \t total loss:{}".format(epoch, loss_total))
            loss_total = 0
torch.save(model.state_dict(), "./model.pth")
