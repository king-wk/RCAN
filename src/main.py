import torch
import utility
import data
import model
import loss
from option import args
from trainer import Trainer


torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    # dataloader
    print("===> Loading data...")
    loader = data.Data(args)
    # model
    model = model.Model(args, checkpoint)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    if not args.test_only:
        print("===> Start Training...")
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

