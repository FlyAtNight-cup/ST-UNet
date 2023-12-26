from .model.factory import create_segmenter

net_kwargs = {'image_size': (512, 512), 
              'patch_size': 16, 
              'd_model': 192, 
              'n_heads': 3, 
              'n_layers': 12,
              'normalization': 'vit', 
              'distilled': False,
              'backbone': 'vit_tiny_patch16_384', 
              'dropout': 0.0, 'drop_path_rate': 0.1, 
              'decoder': {'drop_path_rate': 0.0, 
                          'dropout': 0.1, 
                          'n_layers': 2, 
                          'name': 'mask_transformer', 
                          'n_cls': 7
                          }, 
              'n_cls': 7}

def get_segmenter():
    return create_segmenter(net_kwargs)
