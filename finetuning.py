import torch.optim as optim
import torch
from tqdm.notebook import tqdm
from tabulate import tabulate
import torch.nn.functional as F

from cnn_workflow import cnn_workflow, utils
from copy import deepcopy


def lr_test(device, data_loader, epoch_num, lrs, vgg_builder, note, save_epochs, pre_step, pre_step_sigma):
    """
    train network with different learning rates (@param lrs)
    """
    results = []
    for lr in lrs:
        model  = vgg_builder.build().to(device)
        utils.add_model_note(model, note)
        opt = optim.Adadelta(model.parameters(), lr=lr)
        r = cnn_workflow.train(model, device, data_loader, opt, epoch_num=epoch_num, save_epochs=save_epochs, loss_fn=F.cross_entropy, pre_step=pre_step, pre_step_sigma=pre_step_sigma)

        results.append(r)

    return results

def stochastic_smoothing(model, loss_fn, trn_x, trn_labels, sigma):
    """
    Part 4 of assignment: 
    copy model, add noise to parameters, compute gradients of copied model, set value of original model gradients to value of gradients of copied model
    """
    model_copy = deepcopy(model)

    for p in model_copy.parameters():
        n = torch.normal(mean=torch.zeros_like(p), std=torch.ones_like(p)*sigma)
        p.data+=n

    output = model_copy(trn_x)
    trn_loss = loss_fn(output, trn_labels)
    trn_loss.backward()

    for p1, p2 in zip(model.parameters(), model_copy.parameters()):
        p1.grad.data = p2.grad.data


def get_test_results(filter, data_loader, device, vgg_builder, first_n=-1, va_threshold=1.0):
    """
    Sort training results by validation accuracy and valication loss, test first_n epochs on a test set,
    return first_n results.
    """
    subdirs = utils.filter_subdirs(filter)
    params = utils.load_params_subdirs(subdirs)
    results_list = utils.load_results_subdirs(subdirs)

    data = []
    for results, p in zip(results_list, params):
        for epoch, result in enumerate(results, start=1):
            data.append((p.opt_lr, epoch, result, p.data_dir + f'/models/{p.model_name}_{epoch}.pt'))

    data_best_accuracy = [x for x in data if x[2].val_acc>=va_threshold]
    print(f'{len(data_best_accuracy)}/{len(data)} epochs has validation accuracy >= {va_threshold}')

    data_sorted = sorted(data_best_accuracy, key=lambda x: (1-x[2].val_acc, x[2].val_loss)) # sort by va and vl

    if first_n!=-1:
        data_sorted = data_sorted[:first_n]

    test_results = []
    for idx, (lr, epoch, result, model_state_file) in enumerate(tqdm(data_sorted), start=1):
        model = vgg_builder.build().to(device)
        utils.load_state_to_network(device, model_state_file, model)
        tl, ta = cnn_workflow.evaluate(model, device, data_loader.test_loader, loss_fun=F.cross_entropy)
        test_results.append((idx,lr,epoch,result.val_loss,result.val_acc, tl , ta, model_state_file))


    headers= ['#','lr', 'epoch', "val loss", 'val acc',"test loss", "test acc", 'model path']

    with open(f'test_results/{" ".join(filter)}.txt', 'w') as f:
      f.write(tabulate(test_results, headers=headers))

    return test_results, headers