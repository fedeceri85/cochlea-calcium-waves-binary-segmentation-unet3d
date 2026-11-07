import time
import torch 

def train_one_epoch_binary(model,loader,optimizer,loss_function,device):

    model.train()
    epoch_loss = 0

    #end_step = time.perf_counter()
    for batch_number, batch in enumerate(loader):
        #batch_ready = time.perf_counter()
        #data_wait = batch_ready - end_step

        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(images)

        loss = loss_function(pred,labels)

        loss.backward()
        optimizer.step()

        #torch.cuda.synchronize()
       #end_step = time.perf_counter()
        #compute_time = end_step - batch_ready
        # if batch_number % 10 == 0:
        #     print(
        #         f"data_wait={data_wait:.3f}s, "
        #         f"compute={compute_time:.3f}s"
        #     )
        epoch_loss += loss.item()
    return epoch_loss/len(loader)