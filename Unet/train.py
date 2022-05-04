import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

from settings import *
from utils import save_checkpoint, save_predictions_as_imgs


def train_fn(loader, model, optimizer, loss_fn, scaler=None):
    torch.cuda.empty_cache()
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        if torch.cuda.is_available():

            with torch.cuda.amp.autocast():
                predictions = model(data)
                if targets.shape != predictions.shape:
                    targets = torchvision.transforms.functional.resize(targets, size=predictions.shape[2:])
                loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_postfix(loss=float(loss))
        else:
            preds = model.forward(data)
            loss_value = loss_fn(preds, targets)
            loss_value.backward()

            optimizer.step()
            loop.set_postfix(loss=loss_value.item())

        if batch_idx % (len(loader) // 20) == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)