import math
import torch
import numpy as np
import torchvision
from torchvision import models, transforms


def get_imnet_val_acts(model, valdir, batch_size=16, selected_neuron=None, neuron_coord=None, sort_acts=True, device='cpu'):
    IMAGE_SIZE = 336
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    clip_norm = transforms.Normalize(mean=clip_mean, std=clip_std)

    imagenet_data = torchvision.datasets.ImageFolder(valdir, transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, drop_last=False)

    input_acts = []
    activations = []
    most_images = []
    all_images = []
    act_list = []
    im_count = 0
    for j, (inputs, labels) in enumerate(dataloader):
        with torch.no_grad():
            # print(j)
            im_count += inputs.shape[0]

            inputs, labels = inputs.to(device), labels.to(device)
            norm_inputs = clip_norm(inputs)

            acts = torch.squeeze(model(norm_inputs))
            activations.append(acts.cpu())

            inputs = inputs.cpu()
            for inp in inputs:
                all_images.append(inp)

    all_ord_list = np.arange(im_count).tolist()
    unrolled_act = [num for sublist in activations for num in sublist]
    if sort_acts:
        all_act_list, all_ord_sorted = zip(*sorted(zip(unrolled_act, all_ord_list), reverse=True))
        return all_images, act_list, unrolled_act, all_act_list, all_ord_sorted, input_acts
    else:
        return all_images, act_list, unrolled_act, None, None, input_acts


def validate(model, valdir, batch_size=1024, k=1, scale=1, lesions=None, use_cuda=True):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)

    IMAGE_SIZE = 224

    #   scale = 1 -> no scale
    #   scale > 1 -> crop to scale fraction of 224, then resize to 224
    #   scale < 1 -> crop to 224, then resize to fraction of 224, pad to 224
    transform = None
    if scale == 1:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
    elif scale > 1:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(int(IMAGE_SIZE*(scale-1))),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
    elif scale < 1:
        resize = int(IMAGE_SIZE*scale) if int(IMAGE_SIZE*scale) % 2 == 0 else math.ceil(IMAGE_SIZE*scale)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.Resize(resize),
            transforms.Pad(int((IMAGE_SIZE - resize) / 2), padding_mode='reflect'),
            transforms.ToTensor(),
        ])

    imagenet_data = torchvision.datasets.ImageFolder(valdir, transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, drop_last=False)

    correct = 0
    total = 0
    for j, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()

        norm_inputs = norm_transform(inputs)
        outputs = model(norm_inputs, lesion_chans=lesions)

        #   Compute the top-k predictions
        _, predicted = torch.topk(outputs, k=k, dim=1)

        #   Check if the true label is in the top-k predictions
        correct += (predicted == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    #   Compute the top-k accuracy
    accuracy = 100 * (correct / total)
    print(f'Top {k} accuracy ', accuracy)

    return accuracy