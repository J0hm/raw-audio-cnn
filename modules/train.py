from tqdm import tqdm
from sklearn import metrics
import numpy
import torch

def train(model, transform, criterion, optimizer, scheduler, epoch, loader, device, verbose=False, log_interval=10):
    model.train()

    total_loss = 0
    count = 0
    
    print("Beginning epoch", epoch, "...")

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data = data.to(device)
        target = target.to(device)

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        loss = criterion(output.squeeze(), target) # is squeeze necessary? might not be if there are no dimensions of size 1
        total_loss += loss.item()
        count += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print training stats
        if verbose and batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [({100. * batch_idx / len(loader.dataset):.1f}%)]\tLoss: {loss.item():.6f}\tAvg loss: {total_loss / count:.6f}")

    print(f"Epoch: {epoch} completed\tAvg loss: {total_loss / count:.6f}")
    scheduler.step(total_loss)
    return total_loss / count

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    s, _ = torch.sort(tensor)
    print(s)
    return tensor.argmax(dim=-1)

def test(model, transform, epoch, loader, device, verbose=False, labels=None):
    model.eval()
    correct = 0
    counts_pred = numpy.zeros(35, dtype = int)
    counts_actual = numpy.zeros(35, dtype = int)

    pred_list = []
    actual_list = []

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        
        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        for p in pred:
            counts_pred[p] += 1
            pred_list.append(p.item())
        for l in target:
            counts_actual[l] += 1
            actual_list.append(l.item())
    
    report = metrics.classification_report(actual_list, pred_list, digits=3, target_names=labels, output_dict=True)

    

    if(verbose):
        print(metrics.classification_report(actual_list, pred_list, digits=3, target_names=labels))

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%)\n")
    return correct/len(loader.dataset), report['macro avg']

