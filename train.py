from tqdm import tqdm
import numpy

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
            print(f"Train Epoch: {epoch} [({100. * batch_idx / len(loader):.1f}%)]\tLoss: {loss.item():.6f}\tAvg loss: {total_loss / count:.6f}")

    print(f"Epoch: {epoch} completed\tAvg loss: {total_loss / count:.6f}")
    scheduler.step(total_loss)

def number_of_correct(pred, target):
    # count number of correct predictions
    return pred.squeeze().eq(target).sum().item()

def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)

def test(model, transform, epoch, loader, device, verbose=False):
    model.eval()
    correct = 0
    counts_pred = numpy.zeros(35, dtype = int)
    counts_actual = numpy.zeros(35, dtype = int)

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        
        for l in target:
            counts_actual[l] += 1

        # apply transform and model on whole batch directly on device
        data = transform(data)
        output = model(data)

        pred = get_likely_index(output)
        correct += number_of_correct(pred, target)

        for p in pred:
            counts_pred[p] += 1

    if(verbose):
        print("Predicted label counts:\n", counts_pred)
        print("Actual label counts:\n", counts_actual)
        print("Diff:\n", (counts_pred-counts_actual))

    print(f"\nTest Epoch: {epoch}\tAccuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

