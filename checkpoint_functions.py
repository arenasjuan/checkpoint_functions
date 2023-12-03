import os
import torch
import copy
import safetensors


# Converts .ckpt files to .safetensor and vice versa
def convert_checkpoint(file_path, output_filename=None):
    try:
        if file_path.endswith('.ckpt'):
            weights = torch.load(file_path, map_location=torch.device('cpu'))
            if "state_dict" in weights:
                weights = weights["state_dict"]
            safetensors.torch.save(weights, output_filename if output_filename else file_path.replace('.ckpt', '.safetensors'))

        else:
            weights = safetensors.torch.load(file_path, map_location=torch.device('cpu'))
            torch.save(weights, open(output_filename if output_filename else file_path.replace('.safetensors', '.ckpt'), "wb"))

    except Exception as e:
        print(f"Error converting file: {e}")


# Bake a VAE into a checkpoint of the same type
def bake_VAE(vae_file_path, model_file_path):
    load_function = safetensors.torch.load if vae_file_path.endswith('.safetensors') else torch.load
    vae_model = load_function(vae_file_path, map_location="cpu")
    full_model = load_function(model_file_path, map_location="cpu")

    # Use the full model if 'state_dict' is not a key
    vae_model = vae_model if 'state_dict' not in vae_model else vae_model['state_dict']
    full_model = full_model if 'state_dict' not in full_model else full_model['state_dict']

    for k, v in vae_model.items():
        if k[:4] not in ["loss", "mode"]:
            full_model["first_stage_model." + k] = copy.deepcopy(v)

    output_filename = model_file_path.rsplit('.', 1)[0] + '_BakedVAE' + model_file_path.rsplit('.', 1)[-1]

    save_function = safetensors.torch.save if model_file_path.endswith('.safetensors') else torch.save
    save_function(full_model, output_filename)