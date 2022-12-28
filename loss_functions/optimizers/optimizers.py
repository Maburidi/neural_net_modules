

######### 1 ###########
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.wd,
    )

######### 2 ###########
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),          # only optimize the requires_grad parameters. Here we filter the rest
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            lr=self.lr)




