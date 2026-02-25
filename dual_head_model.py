#!/usr/bin/env python3
"""
Dual-Head NER Model — BIO + Boundary (START/END) heads

Architecture:
    BERT Encoder (768-dim)
          │
    shared hidden states
       /           \
  BIO Head      Boundary Head
  Linear(768,3) Linear(768,3)
  O/B-TERM/I-TERM  O/START/END

Loss: total = bio_loss + alpha * boundary_loss
Inference: BIO primary, boundary for optional rescue
"""

from transformers import BertConfig, BertPreTrainedModel, BertModel
import torch
import torch.nn as nn


class DualHeadConfig(BertConfig):
    """Extended BertConfig with dual-head NER parameters."""
    model_type = "bert"

    def __init__(
        self,
        bio_num_labels: int = 3,
        boundary_num_labels: int = 3,
        boundary_loss_alpha: float = 0.3,
        dual_head: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bio_num_labels = bio_num_labels
        self.boundary_num_labels = boundary_num_labels
        self.boundary_loss_alpha = boundary_loss_alpha
        self.dual_head = dual_head
        # Override num_labels for compatibility with HF ecosystem
        self.num_labels = bio_num_labels


class BertForDualHeadNER(BertPreTrainedModel):
    """BERT with two classification heads: BIO tagging + boundary detection."""
    config_class = DualHeadConfig

    def __init__(self, config: DualHeadConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bio_classifier = nn.Linear(config.hidden_size, config.bio_num_labels)
        self.boundary_classifier = nn.Linear(config.hidden_size, config.boundary_num_labels)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        boundary_labels=None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)

        bio_logits = self.bio_classifier(sequence_output)
        boundary_logits = self.boundary_classifier(sequence_output)

        return {
            "bio_logits": bio_logits,
            "boundary_logits": boundary_logits,
        }

    @classmethod
    def from_pretrained_bert(
        cls,
        bert_model_name: str,
        bio_num_labels: int = 3,
        boundary_num_labels: int = 3,
        boundary_loss_alpha: float = 0.3,
        id2label: dict = None,
        label2id: dict = None,
    ):
        """Initialize from a vanilla pretrained BERT model.

        Loads BERT encoder weights from the given model name,
        and randomly initializes the two classification heads.
        """
        config = DualHeadConfig.from_pretrained(
            bert_model_name,
            bio_num_labels=bio_num_labels,
            boundary_num_labels=boundary_num_labels,
            boundary_loss_alpha=boundary_loss_alpha,
            dual_head=True,
        )
        if id2label is not None:
            config.id2label = id2label
        if label2id is not None:
            config.label2id = label2id

        model = cls(config)

        # Load pretrained BERT encoder weights
        pretrained = BertModel.from_pretrained(bert_model_name)
        model.bert.load_state_dict(pretrained.state_dict(), strict=False)
        del pretrained

        return model
