from typing import List
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaLayer
from ...utils.modeling import freeze
from . import ModalityModel

class PlainTransformer(ModalityModel):
    def __init__(self, arch_name: str, num_classes: int, freeze_strategy: str = "no-freeze", **kwargs):
        super().__init__()
        assert freeze_strategy in ["freeze", "gradual-unfreeze", "no-freeze"]
        self.model = AutoModelForSequenceClassification.from_pretrained(
            arch_name, num_labels=num_classes
        )
        if freeze_strategy == "freeze":
            # freeze base lm
            for n, c in self.model.named_children():
                if n != "classifier":
                    freeze(c)
        self.tokenizer = AutoTokenizer.from_pretrained(
            arch_name
        )
        self.max_len = self.tokenizer.model_max_length
    
    def forward(self, input_ids, attention_mask, **kwargs):
        """
        if not eval mode, use forward template
          in `bimodule` can customize this forward
          NOTE: due to dropout, forward twice can get different output
        """
        if not self.training:
            output = self.model(input_ids, attention_mask=attention_mask)
            return output.logits
        block_input, extendend_attention_mask = \
            self.before_layers(input_ids, attention_mask)
        for block in self.blocks:
            layer_outputs = self.block(block,
                block_input, extendend_attention_mask)
            block_input = layer_outputs[0]
        logits = self.after_layers(block_input)
        return logits
    
    def before_layers(self, input_ids, attention_mask):
        """
        prepare everything before feeding into layers (in encoder)
        """
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        buffered_token_type_ids = self.model.roberta.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded

        embedding_output = self.model.roberta.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        return embedding_output, self.model.roberta.get_extended_attention_mask(attention_mask, input_shape, input_ids.device)
    
    def block(self, block: RobertaLayer, 
            hidden_states, attention_mask, skip_input=None):
        """
        @skip_input: from other modality
        """
        self_attention_outputs = block.attention(
            hidden_states, attention_mask)
        attention_output = self_attention_outputs[0]
        intermediate_output = block.intermediate(attention_output)
        if skip_input is None:
            layer_output = block.output(intermediate_output, attention_output)
        else:
            hidden_states = block.output.dropout(block.output.dense(intermediate_output))
            layer_output = block.output.LayerNorm(hidden_states + skip_input)
        return (layer_output, )

    def after_layers(self, encoder_outputs):
        logits = self.model.classifier(encoder_outputs)
        return logits

    def tokenize(self, sents: List[str]):
        return self.tokenizer(
            sents, padding=True, truncation=True, 
            max_length=self.max_len, return_tensors="pt")
    
    def replace_linear(self, cls: nn.Module, num_out: int):
        self.model.classifier.out_proj = cls(
            self.model.classifier.out_proj.in_features, num_out
        )

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def blocks(self) -> List[RobertaLayer]:
        return self.model.roberta.encoder.layer