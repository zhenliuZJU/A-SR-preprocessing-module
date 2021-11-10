from utils.utils import *
from net import *
from dataloader import *
import numpy as np

import os

os.environ['CUDA_ENABLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    training_data_loader = dataloaders

    print("===> Building model")
    if opt.SRNet == 'FSRCNN':
        SRmodel = FSRCNN()
    else:
        SRmodel = VDSR()

    CNET = SDCNN()
    criterion = nn.CrossEntropyLoss()

    print("===> Setting GPU")
    SRmodel = SRmodel.cuda()
    criterion = criterion.cuda()
    CNET = CNET.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            SRmodel.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained_SR:
        if os.path.isfile(opt.pretrained_SR):
            print("=> loading SR model '{}'".format(opt.pretrained_SR))
            weights = torch.load(opt.pretrained_SR)
            SRmodel.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained_SR))

    if opt.pretrained_TL:
        if os.path.isfile(opt.pretrained_TL):
            print("=> loading TL model '{}'".format(opt.pretrained_TL))
            weights = torch.load(opt.pretrained_TL)
            # CNET.load_state_dict(weights)
            CNET.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained_TL))

    print("===> Setting Optimizer")
    optimizer_SR = optim.SGD(SRmodel.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    # optimizer_TL = optim.SGD(vgg16.fc.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    # optimizer_TL = optim.SGD(vgg16.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer_TL = optim.SGD(CNET.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(dataloaders, optimizer_SR, optimizer_TL, SRmodel, CNET, criterion, epoch)
        if opt.SR_used:
            save_SR_checkpoint(SRmodel, epoch)
        if opt.TLtrain:
            save_TL_checkpoint(CNET, epoch)


def train(training_data_loader, optimizer_SR, optimizer_TL, SRmodel, CNET, criterion, epoch):
    lr = adjust_learning_rate(optimizer_SR, optimizer_TL, epoch - 1)

    for param_group in optimizer_SR.param_groups:
        param_group["lr"] = lr

    for param_group in optimizer_TL.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer_SR.param_groups[0]["lr"]))

    # total = 0
    # count = 0
    # false = 0

    SRmodel.train()

    total = 0

    for iteration, batch in enumerate(training_data_loader['train'], 1):

        images, labels = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        images = images.cuda()
        labels = labels.cuda()

        if opt.SR_used:
            out_images = SRmodel(images)
            output = CNET(out_images)
        else:
            output = CNET(images)

        predicted = output
        # total += labels.size(0)
        # false += (torch.max(predicted.cpu(),1) != labels).sum()
        # count = count+1

        loss = criterion(predicted, labels)
        if opt.SRtrain:
            optimizer_SR.zero_grad()
        if opt.TLtrain:
            optimizer_TL.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(SRmodel.parameters(), opt.clip)
        nn.utils.clip_grad_norm_(CNET.parameters(), opt.clip)

        if opt.SRtrain:
            optimizer_SR.step()
        if opt.TLtrain:
            optimizer_TL.step()

        total += loss.item()

        if iteration % 100 == 0:
            print("===> Average Loss: {:.10f}".format(total / iteration))

    if opt.SR_used:
        # Get a batch of training data
        out_images, classes = out_images.cpu(), labels.cpu()
        inputs = images.cpu()

        # Make a grid from batch
        out_images = torchvision.utils.make_grid(out_images)
        inp = torchvision.utils.make_grid(inputs)

        imshow(out_images, title=[class_names[x] for x in classes])
        imshow(inp, title=[class_names[x] for x in classes])


if __name__ == "__main__":
    main()