from PIL import Image
import imageio
import matplotlib.pyplot as plt
import cv2

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def save_SR_checkpoint(SRmodel, epoch):
    model_out_path = "./checkpoint/" + "model_FSRCNN_CNET_*8_epoch_{}.pth".format(
        epoch + opt.pretrained_SR_num)
    state = {"epoch": epoch, "model": SRmodel}
    if not os.path.exists("./checkpoint/"):
        os.makedirs("./checkpoint/")

    torch.save(state, model_out_path)

    print("Super resolution network checkpoint saved to {}".format(model_out_path))


def save_TL_checkpoint(model, epoch):
    model_out_path = "./checkpoint/TransferLearning/" + "model_FSRCNN_CNET_*8_epoch_{}.pth".format(
        epoch + opt.pretrained_TL_num)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists("./checkpoint/TransferLearning/"):
        os.makedirs("./checkpoint/TransferLearning/")

    torch.save(state, model_out_path)

    print("Transfer learning checkpoint saved to {}".format(model_out_path))


def adjust_learning_rate(optimizer_SR, optimizer_TL, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr
