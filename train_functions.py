import math
import torch
import os
import numpy as np
try:
    from IPython.display import clear_output
except:
    pass
import torch.nn.functional

# TODO per class accuracy saving

def _progress_bar(value,
                  length=40,
                  title=" ",
                  vmin=0.0,
                  vmax=1.0,
                  show_numeric=True,
                  batch_size=0):
    """
    Text _progress_bar bar
    Parameters
    ----------
    value : float
        Current value to be displayed as _progress_bar
    vmin : float
        Minimum value
    vmax : float
        Maximum value
    length: int
        Bar length (in character)
    title: string
        Text to be prepend to the bar
    show_numeric: bool
         True if you want how many batches processed.
    batch_size: int
        How many items in batch. Only useful if show_numric is True. If zero, then not shown in output. Must be equal or bigger than zero.

    """
    assert batch_size >= 0

    progress_value = value

    # Block progression is 1/8
    blocks = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    vmin = vmin or 0.0
    vmax = vmax or 1.0
    lsep, rsep = "▏", "▕"

    # Normalize value
    value = min(max(value, vmin), vmax)
    value = (value - vmin) / float(vmax - vmin)

    v = value * length
    x = math.floor(v)  # integer part
    y = v - x  # fractional part
    base = 0.125  # 0.125 = 1/8
    prec = 3
    i = int(round(base * math.floor(float(y) / base), prec) / base)
    bar = "█" * x + blocks[i]
    n = length - len(bar)
    bar = lsep + bar + " " * n + rsep

    base_output = title + bar + " %.1f%%" % (value * 100)

    if show_numeric:
        base_output += f" | Batches: {progress_value}/{int(vmax)}"
        if batch_size != 0:
            base_output += f" | Items: {progress_value*batch_size}/{int(vmax*batch_size)}"

    return base_output


def _evaluate_model(model,
                    dataloader,
                    loss_fn,
                    device=torch.device("cpu"),
                    ):
    model.eval()

    loss_array = []

    with torch.no_grad():
        for (X, Y) in dataloader:
            X = torch.transpose(X.to(device), 1, -1)
            Y = Y.to(device)
            pred = model(X)

            loss = loss_fn(pred, Y)
            loss_array.append(loss.item())

    return np.array(loss_array), np.average(loss_array)


def _epoch_train(model,
                 train_dataloader,
                 optimizer,
                 epoch,
                 loss_fn,
                 device=torch.device("cpu"),
                 update_every_n_batches=100,
                 conclusions=None,
                 ):
    model.train()

    # backprop loss, reconstruction loss, kld loss
    loss_array = []

    size = len(train_dataloader.dataset)
    for batch, (X, Y) in enumerate(train_dataloader):
        X = torch.transpose(X.to(device), 1, -1)
        Y = Y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_array.append(loss.item())

        if batch % update_every_n_batches == 0:
            current = batch * len(X)

            os.system("clear")
            try:
                clear_output(wait=True)
            except:
                pass
            print("Previous:")
            for conclusion in conclusions:
                print(conclusion)
            print("\nCurrent:")

            print(f"Epoch: {epoch+1:>3d}")
            print(f"{_progress_bar(batch, vmin=0, vmax=train_dataloader.__len__(), title='Training Progress: ')} | Mean training loss: {np.mean(loss_array)} | Training loss: {loss.item():.5f} | Items: {current:>5d}/{size:>5d}")

    return np.array(loss_array)


def train_with_epoches(model,
                       max_epoches,
                       train_dataloader,
                       optimizer,
                       loss_fn,
                       evaluate_dataloader=None,
                       device=torch.device("cpu"),
                       save_model_each_epoch=True,
                       model_name="Models\\MyModel",
                       update_every_n_batches=100):
    conclusions = []

    # statistics

    epoched_training_loss_array = []
    epoched_testing_loss_array = []
    epoched_average_testing_loss = []

    average_testing_loss = 0

    model.to(device)

    for epoch in range(max_epoches):
        training_loss = _epoch_train(model, train_dataloader, optimizer, epoch, loss_fn, device, update_every_n_batches, conclusions)
        epoched_training_loss_array.append(training_loss)

        if evaluate_dataloader is not None:
            testing_loss, average_testing_loss = _evaluate_model(model, evaluate_dataloader, loss_fn, device)
            epoched_testing_loss_array.append(testing_loss)
            epoched_average_testing_loss.append(average_testing_loss)

            conclusions.append(f"{epoch+1} Epoch end | Training loss: {np.average(training_loss):.5f} | Testing loss: {average_testing_loss:.5f}")

        if save_model_each_epoch:
            try:
                torch.save(model.state_dict(), f"{model_name}__epoch__{epoch+1}__train_loss__{np.average(training_loss):.5f}{f'__test_loss_{average_testing_loss:.5f}' if evaluate_dataloader is not None else ''}")
            except:
                os.makedirs(model_name)
                os.rmdir(model_name)

                torch.save(model.state_dict(), f"{model_name}__epoch__{epoch+1}__train_loss__{np.average(training_loss):.5f}{f'__test_loss_{average_testing_loss:.5f}' if evaluate_dataloader is not None else ''}")

    os.system("clear")
    try:
        clear_output(wait=True)
    except:
        pass
    print("Training finished.\nAll runs:")
    for conclusion in conclusions:
        print(conclusion)

    return np.array(epoched_training_loss_array), \
           np.array(epoched_testing_loss_array), \
           np.array(epoched_average_testing_loss)