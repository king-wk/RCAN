from importlib import import_module
from torch.utils.data import dataloader


class Data:
    def __init__(self, args):

        self.loader_train = None
        if not args.test_only:
            module_train = import_module('data.' + args.data_train.lower())
            # dataset
            trainset = getattr(module_train, args.data_train)(args)
            self.loader_train = dataloader.DataLoader(
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=not args.cpu,
                num_workers=args.n_threads,
            )

        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100', 'test_1080', 'test_2160']:
            module_test = import_module('data.benchmark')
            testset = getattr(module_test, 'Benchmark')(args, train=False, name=args.data_test)
        else:
            module_test = import_module('data.' +  args.data_test.lower())
            testset = getattr(module_test, args.data_test)(args, train=False, name=args.data_test)

        self.loader_test = dataloader.DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=args.n_threads,
        )
