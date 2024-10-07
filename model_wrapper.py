import torch
from torch import nn


#   Wrapper so that we have control over the forward pass.
class ModelWrapper(nn.Module):

    target_neuron = None
    all_acts = []

    def __init__(self, model, input_ids, image_processor, layer_number=None, use_sae=False, expansion=8):
        super().__init__()
        self.model = model
        self.image_processor = image_processor

        self.input_ids = None
        if input_ids is not None:
            self.input_ids = input_ids[None, :]

        self.vision_tower = model.model.vision_tower
        self.projector = model.model.mm_projector

        self.llm = nn.Sequential()
        self.mlp = nn.Sequential()
        for i in range(layer_number):
            self.llm.append(model.model.layers[i])
        
        self.use_sae = use_sae
        if self.use_sae:
            self.map = nn.Linear(4096, 4096*expansion, bias=False)


    @torch.autocast(device_type="cuda")
    def forward(self, x):
        pos_ids = None
        attn_mask = None
        extract_pos = None

        # x = self.image_processor.preprocess(x, do_rescale=False, return_tensors='pt')['pixel_values']
        if self.input_ids is not None:
            pos_ids = torch.arange(0, self.input_ids.shape[1], dtype=torch.long, device=self.input_ids.device)
            pos_ids = torch.broadcast_to(pos_ids, (x.shape[0], pos_ids.shape[0]))
            input_tokens = torch.broadcast_to(self.input_ids, (x.shape[0], self.input_ids.shape[1]))
            _, pos_ids, attn_mask, _, x, _ = self.model.prepare_inputs_labels_for_multimodal(input_tokens, pos_ids, None, None, None, x)
            extract_pos = (x.shape[1] // 2) + input_tokens.shape[1]
        else:
            x = self.vision_tower(x)
            x = self.projector(x)
            pos_ids = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)[None, :]
            attn_mask = torch.ones((x.shape[0], 1, x.shape[1], x.shape[1]), dtype=torch.bool, device=x.device)
            extract_pos = x.shape[1] // 2

        for l in self.llm:
            x = l(x, position_ids=pos_ids, attention_mask=attn_mask)[0]

        # x = x[:, extract_pos, :]
        # x = x[:, -1, :]
        # x = torch.mean(x, dim=1)

        if self.use_sae:
            x = self.map(x)

        # print(x[0, self.target_neuron])

        x = torch.max(x, dim=1)[0]

        if self.target_neuron is not None:
            self.all_acts.append(x[0, self.target_neuron].detach().cpu().item())

        #   add empty spatial dimensions for lucent's activation extraction
        x = x[:, :, None, None]

        return x