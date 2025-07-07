import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize
from .attention import AttBlockLKA
from .layers import AbundanceScaling, SumToOne, Nonnegative

class StructuredAutoencoder(nn.Module):
    def __init__(self, input_channels, num_abundance_maps, filters_per_map):
        super().__init__()
        self.pruned_layers = []
        self.indices_to_keep = list(range(num_abundance_maps))
        self.input_channels = input_channels
        self.num_abundance_maps = num_abundance_maps
        self.filters_per_map = filters_per_map
        self.asc_layer = SumToOne(0.5)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, input_channels*4, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(input_channels*4),
            nn.ELU(),
            AttBlockLKA(dim=input_channels*4),
            nn.ELU(),
            nn.Conv2d(input_channels*4, num_abundance_maps * filters_per_map, kernel_size=3, 
                     padding=1, padding_mode='reflect', groups=input_channels*4),
            nn.BatchNorm2d(num_abundance_maps * filters_per_map),
            nn.ReLU()
        )

        self.abundance_layer = nn.Conv2d(num_abundance_maps * filters_per_map, 
                                       num_abundance_maps, kernel_size=1,
                                       bias=False, padding_mode='reflect')
        
        self.scaling_matrix = AbundanceScaling(num_abundance_maps)
        self.decoder = nn.Conv2d(num_abundance_maps, input_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.05)

        nn.init.uniform_(self.abundance_layer.weight, 0., 1.)
        parametrize.register_parametrization(self.scaling_matrix, "scaling_matrix", Nonnegative())
        parametrize.register_parametrization(self.decoder, "weight", Nonnegative())

    def forward(self, x):
        encoded = self.encoder(x)
        abundance_pre = self.abundance_layer(encoded)
        abundance_maps = self.asc_layer(abundance_pre, self.indices_to_keep)
        abundance_maps_scaled = self.scaling_matrix(abundance_maps)
        abundance_maps_scaled = self.dropout(abundance_maps_scaled)
        decoded = self.decoder(abundance_maps_scaled)
        return decoded, abundance_maps