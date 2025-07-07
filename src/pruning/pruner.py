import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.nn.utils.parametrize as parametrize
from ..models.layers import Nonnegative
from ..utils.metrics import sad_decoder
import copy

def analyze_abundance_maps(model, input_data, pruned_layers):
    model.eval()
    with torch.no_grad():
        _, abundance_maps = model(input_data)
        endmembers = copy.deepcopy(model.decoder.weight)

    gram = pairwise_difference(abundance_maps, endmembers, pruned_layers)
    gram_sum = torch.sum(gram, dim=(0,1))
    batch, abd1, abd2 = torch.unravel_index(torch.argmax(gram), gram.shape)

    print('Most corrolated maps:', abd1.item() + 1, ' and ', abd2.item() + 1 )
    if torch.norm(abundance_maps[:, abd1, :, :]).item() < torch.norm(abundance_maps[:, abd2, :, :]).item():
        prune_index = abd1.item()
    else:
        prune_index = abd2.item()

    #print(prune_index + 1)
    #tmp = input('Which map to prune ?')
 
    #if tmp in range(0,13):
    #    print(tmp)
    #    prune_index = tmp - 1 

    #print(prune_index + 1)

    metrics = []
    for i in range(abundance_maps.size(1)):
        if i not in pruned_layers:
            map_data = abundance_maps[:, i, :, :]
            metrics.append({
                'map_number': i + 1,
                'L2_norm': torch.norm(map_data).item(),
                'L1_norm': torch.norm(map_data, p = 1).item(),
                'abund_norm_div': 100 * torch.norm(map_data).item() / torch.count_nonzero(map_data).item(),
                'decoder_norm': torch.norm(model.decoder.weight[:,i,0,0]).item(),
            })

    return metrics, prune_index

def prune_abundance_map(model, map_index, pruned_layers):
    mask = torch.ones_like(model.abundance_layer.weight)
    if map_index:
        mask[map_index, :, :, :] = 0
    mask[pruned_layers, :, :, :] = 0
    prune.CustomFromMask.apply(model.abundance_layer, 'weight', mask)

def prune_decoder(model, map_index, pruned_layers):
    if parametrize.is_parametrized(model.decoder):
        try:
            parametrize.remove_parametrizations(model.decoder, 'weight')
        except:
            parametrize.remove_parametrizations(model.decoder, 'weight_orig')

    weight_tensor = getattr(model.decoder, 'weight')
    mask = torch.ones_like(weight_tensor)
    mask[:, pruned_layers] = 0

    if map_index:
        mask[:, map_index] = 0

    prune.CustomFromMask.apply(model.decoder, 'weight', mask)
    parametrize.register_parametrization(model.decoder, 'weight_orig', Nonnegative())

def pairwise_difference(abundance_maps, endmembers, pruned_layers, kernel_size=5, sigma=5.):
    B, N, H, W = abundance_maps.shape
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    x, y = torch.meshgrid(coords, coords)
    kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel[kernel_size//2, kernel_size//2] = 1
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(abundance_maps.device)

    pairwise_uniqueness = torch.zeros(B, N, N, device=abundance_maps.device)
    pad_size = kernel_size // 2

    for i in range(N):
        for j in range(i+1, N):
            if i not in pruned_layers or j not in pruned_layers:
                padded_i = F.pad(abundance_maps[:, i:i+1], (pad_size, pad_size, pad_size, pad_size), 
                               mode='reflect').clone()
                padded_j = F.pad(abundance_maps[:, j:j+1], (pad_size, pad_size, pad_size, pad_size), 
                               mode='reflect').clone()

                local_i = F.conv2d(padded_i, kernel)[0]
                local_j = F.conv2d(padded_j, kernel)[0]

                hadamard = local_i * local_j
                sum2 = torch.abs(torch.sum(local_i) - torch.sum(local_j))
                score = hadamard.sum(dim=(1, 2))
                sad_endmembers = sad_decoder(endmembers[:,i,:,:], endmembers[:,j,:,:])

                denom = torch.norm(abundance_maps[:, i]) * torch.norm(abundance_maps[:, j] * sum2) + 1e-8
                value = 100 * (score)/denom + 0.5*sad_endmembers

                pairwise_uniqueness[:, i, j] = value
                pairwise_uniqueness[:, j, i] = value
            else:
                pairwise_uniqueness[:, i, j] = 0
                pairwise_uniqueness[:, j, i] = 0

    return pairwise_uniqueness