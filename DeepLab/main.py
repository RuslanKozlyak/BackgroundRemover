
from torch import optim, nn

from train import *
from model import test, DeepLab
from utils import load_checkpoint, check_accuracy, save_predictions_as_imgs, save_checkpoint, get_masked_image
import settings


def main():
    model = DeepLab().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
        check_accuracy(val_loader, model, device=DEVICE)

    if torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, device=DEVICE)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE)

if __name__ == '__main__':
    get_masked_image(r"D:\datasets\people_segmentation\images\female-young-woman-beautiful-48196.jpg", 0.43)
    # main()
    # test()
