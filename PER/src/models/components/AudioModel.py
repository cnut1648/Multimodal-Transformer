import os
from typing import Optional
from simplejson import OrderedDict
import torch
from torch import nn
import numpy as np
from einops import rearrange
from typing import List
from openspeech.models.conformer.model import ConformerEncoder
from transformers import AutoModelForSequenceClassification
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor

from transformers.models.hubert.modeling_hubert import HubertForSequenceClassification, HubertEncoderLayerStableLayerNorm


class AudioModel(nn.Module):
    @property
    def hidden_size(self) -> int:
        pass
    @property
    def blocks(self) -> List[nn.Module]:
        """
        normalization blocks
        """
        pass
    def replace_linear(self, cls: nn.Module, num_out: int):
        """
        ordinal regression, replace last linear
        """
        pass

class Conformer(AudioModel):
    def __init__(self, num_classes, input_dim=80, encoder_dim=512, 
            num_layers: int = 17,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            pretrain_ckpt: Optional[str] = None
    ):
        """
        pretrain_ckpt: ckpt of ConformerLSTM from openspeech
        """
        super().__init__()
        self.backbone = ConformerEncoder(
            # num_classes not used unless joint_ctc_attention=True
            num_classes, input_dim=input_dim, encoder_dim=encoder_dim, 
            num_layers= num_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            joint_ctc_attention=False,
        )
        if pretrain_ckpt is not None:
            assert os.path.exists(pretrain_ckpt)
            ckpt = torch.load(pretrain_ckpt)
            old_state_dict = ckpt["state_dict"]
            state_dict = OrderedDict()
            for layer, value in old_state_dict.items():
                if layer.startswith("encoder."):
                    state_dict[layer[8:]] = value
            self.backbone.load_state_dict(state_dict)
        self.out = nn.Linear(encoder_dim, num_classes)

    def forward(self, audios, audio_lengths):
        """
        audios: must be filter bank
        """
        # (b, max_seq, d)
        encoder_outputs, encoder_logits, output_lengths = self.backbone(audios, audio_lengths)
        encoder_outputs = encoder_outputs.mean(1)
        return self.out(encoder_outputs)
    

class HuBERT(AudioModel):
    def __init__(self, pretrain_path, num_classes, sample_rate=16_000):
        super().__init__()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-large-superb-er")
        # self.model = HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ll60k")
        # self.model = HubertForSequenceClassification.from_pretrained("facebook/hubert-large-ll60k")
        # model2 = HubertForSequenceClassification.from_pretrained("superb/hubert-large-superb-er")
        # self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrain_path)
        self.model = HubertForSequenceClassification.from_pretrained(pretrain_path, num_labels=num_classes)
        # self.model.classifier = nn.Linear(256, num_classes)
        self.sample_rate = sample_rate
    
    def forward(self, audios, **kwargs):
        """
        audios: must be raw wave
        """
        inputs = self.feature_extractor(
            audios, sampling_rate=self.sample_rate, return_tensors="pt",
            padding=True
        ).to(self.model.device)
        if not self.training:
            logits = self.model(**inputs).logits
            return logits
        block_input, extendend_attention_mask = self.before_layers(inputs)
        for i, block in enumerate(self.blocks):
            dropout_probability = np.random.uniform(0, 1)
            skip_the_layer = True if dropout_probability < self.model.config.layerdrop else False
            if i == len(self.blocks) - 1 or i == 1:
                skip_the_layer = False
            # skip_the_layer = False
            if not skip_the_layer:
                layer_outputs = self.block(block,
                    block_input, extendend_attention_mask)
                # (bsz, seq_len, d)
                block_input = layer_outputs[0]
        logits = self.after_layers(block_input, inputs["attention_mask"])
        return logits
    
    def before_layers(self, inputs):
        input_values, attention_mask = inputs["input_values"], inputs["attention_mask"]
        extract_features = self.model.hubert.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        attention_mask = self.model.hubert._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        hidden_states = self.model.hubert.feature_projection(extract_features)
        hidden_states = self.model.hubert._mask_hidden_states(hidden_states, mask_time_indices=None)

        hidden_states[~attention_mask] = 0
        # extend attention_mask
        attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)) * -10000.0
        attention_mask = attention_mask.expand(
            attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
        )
        position_embeddings = self.model.hubert.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.model.hubert.encoder.dropout(hidden_states)
        return hidden_states, attention_mask

    def block(self, block: HubertEncoderLayerStableLayerNorm, hidden_states, 
        attention_mask, skip_input=None):
        """
        @skip_input: from other modality
        """
        if skip_input is None:
            attn_residual = hidden_states
        else:
            attn_residual = skip_input
        hidden_states = block.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = block.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=False
        )
        hidden_states = block.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + block.feed_forward(
            block.final_layer_norm(hidden_states))

        outputs = (hidden_states,)
        return outputs

    def after_layers(self, encoder_outputs, attention_mask):
        encoder_outputs = self.model.hubert.encoder.layer_norm(encoder_outputs)
        # (bsz, seq_len, d1 -> d2)
        hidden_states = self.model.projector(encoder_outputs)
        
        padding_mask = self.model._get_feature_vector_attention_mask(
            hidden_states.shape[1], attention_mask)
        hidden_states[~padding_mask] = 0.0
        # (bsz, d2)
        pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        logits = self.model.classifier(pooled_output)
        return logits
    
    @property
    def hidden_size(self) -> int:
        return 1024

    @property
    def blocks(self) -> List[HubertEncoderLayerStableLayerNorm]:
        """
        normalization blocks
        """
        return self.model.hubert.encoder.layers