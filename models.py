import torch
import torch.nn as nn
from functools import partial
from torch.hub import load_state_dict_from_url
import timm.models.swin_transformer as swin
from timm.models.helpers import load_state_dict

class ArkSwinTransformer(swin.SwinTransformer):
    def __init__(self, num_classes_list, projector_features=None, use_mlp=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes_list is not None
        
        self.projector = None
        if projector_features:
            encoder_features = self.num_features
            self.num_features = projector_features
            if use_mlp:
                self.projector = nn.Sequential(
                    nn.Linear(encoder_features, self.num_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.num_features, self.num_features)
                )
            else:
                self.projector = nn.Linear(encoder_features, self.num_features)
        
        # Multi-task heads
        self.omni_heads = nn.ModuleList([
            nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
            for num_classes in num_classes_list
        ])

    def forward(self, x, head_n=None):
        x = self.forward_features(x)
        if self.projector:
            x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]

    def generate_embeddings(self, x, after_proj=True):
        x = self.forward_features(x)
        if after_proj and self.projector:
            x = self.projector(x)
        return x

# modified to load PEAC pretrained weights: https://github.com/jlianglab/PEAC
def build_omni_model_from_checkpoint(args, num_classes_list, key='student'):
    # load PEAC pretrained weights (they have a special swin config)
    # NOTE: we discard the teacher from PEAC and initialize the teacher of Ark to random...
    if args.init == "peac:"
        # PEAC uses swin_base
        if args.model_name == "swin_base":
            model = ArkSwinTransformer(
                num_classes_list,
                args.projector_features,
                args.use_mlp,
                patch_size=4,
                window_size=7,
                embed_dim=128,
                depths=(2, 2, 18, 2),
                num_heads=(4, 8, 16, 32)
            )
        else:
            print("--- PEAC only configured for swin_base ---")
            raise ValueError(f"unknown model name: {args.model_name}")

        if args.pretrained_weights is not None:
            checkpoint = torch.load(args.pretrained_weights, map_location='cpu')
            state_dict = checkpoint[key]
            
            # extract backbone weights from PEAC state_dict
            backbone_state_dict = {}
            for k, v in state_dict.items():
                # only keep the swin weights
                if k.startswith('module.swin_model.'):
                    new_key = k.replace('module.swin_model.', '')
                    backbone_state_dict[new_key] = v

            # discard 'attn_mask' and 'head' keys
            exclude_keys = [k for k in backbone_state_dict.keys() if 'attn_mask' in k or 'head' in k]
            for k in exclude_keys:
                del backbone_state_dict[k]

            # load the backbone weights into the model
            msg = model.load_state_dict(backbone_state_dict, strict=False)
            print(f'Loaded with msg: {msg}')
            print(f'missing keys: {msg.missing_keys}')
            print(f'unexpected keys: {msg.unexpected_keys}')
        
        return model
    else:
        print("--- NOT initializing with PEAC weights ---")
        if args.model_name == "swin_base": #swin_base_patch4_window7_224
            model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
        
        if args.pretrained_weights is not None:
            checkpoint = torch.load(args.pretrained_weights)
            state_dict = checkpoint[key]
            if any([True if 'module.' in k else False for k in state_dict.keys()]):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}

            msg = model.load_state_dict(state_dict, strict=False)
            print('Loaded with msg: {}'.format(msg))     
               
        return model

def build_omni_model(args, num_classes_list):
    if args.model_name == "swin_base": #swin_base_patch4_window7_224
        model = ArkSwinTransformer(num_classes_list, args.projector_features, args.use_mlp, patch_size=4, window_size=7, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))

    if args.pretrained_weights is not None:
        if args.pretrained_weights.startswith('https'):
            state_dict = load_state_dict_from_url(url=args.pretrained_weights, map_location='cpu')
        else:
            state_dict = load_state_dict(args.pretrained_weights)
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
      
        msg = model.load_state_dict(state_dict, strict=False)
        print('Loaded with msg: {}'.format(msg))

    return model

def save_checkpoint(state,filename='model'):
    torch.save(state, filename + '.pth.tar')
