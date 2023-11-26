import torch
import numpy as np
import tinycudann as tcnn


def get_encoder(encoding, input_dim=3, n_bins=16,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512):   
    # Sparse grid encoding
    if "hash" in encoding.lower() or "tiled" in encoding.lower():
        # print("Hash size", log2_hashmap_size)
        per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (n_levels - 1))  # b
        embed = tcnn.Encoding(
            n_input_dims=input_dim,  # 3
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,  # 16 (L)
                "n_features_per_level": level_dim,  # 2 (F)
                "log2_hashmap_size": log2_hashmap_size,  # 16, max hash table size (exp, T)
                "base_resolution": base_resolution,  # 16, N_min
                "per_level_scale": per_level_scale  # b
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims
    
    # Frequency encoding
    elif "freq" in encoding.lower():
        # print("Use frequency")
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                    "otype": "Frequency",
                    "n_frequencies": n_bins
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    # Identity encoding
    elif "identity" in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                    "otype": "Identity"
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    return embed, out_dim