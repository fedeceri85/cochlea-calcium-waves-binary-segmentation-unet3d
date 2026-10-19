def train_one_epoch_binary(model,loader,optimizer,loss_function,device):

    model.train()
    epoch_loss = 0

    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        pred = model(images)

        loss = loss_function(pred,labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss/len(loader)