from torch import nn
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
from ITHP import ITHP
import global_configs
from global_configs import DEVICE


class ITHP_DebertaModel(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (
            global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM, global_configs.VISUAL_DIM
        )

        self.pooler = BertPooler(config)
        self.model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base").to(DEVICE)

        ITHP_args = {
            'X0_dim': TEXT_DIM,
            'X1_dim': ACOUSTIC_DIM,
            'X2_dim': VISUAL_DIM,
            'B0_dim': multimodal_config.B0_dim,
            'B1_dim': multimodal_config.B1_dim,
            'inter_dim': multimodal_config.inter_dim,
            'max_sen_len': multimodal_config.max_seq_length,
            'drop_prob': multimodal_config.drop_prob,
            'p_beta': multimodal_config.p_beta,
            'p_gamma': multimodal_config.p_gamma,
            'p_lambda': multimodal_config.p_lambda,
        }

        self.ITHP = ITHP(ITHP_args)
        self.expand = nn.Linear(multimodal_config.B1_dim, TEXT_DIM)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.beta_shift = multimodal_config.beta_shift

        self.init_weights()

    def forward(self, input_ids, visual, acoustic, attention_mask=None, epoch=0, max_epochs=100):
        """
        前向传播
        
        参数:
            input_ids: 文本input ids
            visual: 视觉特征
            acoustic: 声学特征
            attention_mask: 注意力mask (可选)
            epoch: 当前训练轮次 (用于temperature衰减)
            max_epochs: 最大训练轮次
        """
        embedding_output = self.model(input_ids, attention_mask=attention_mask)
        x = embedding_output[0]  # token-level 表征 [batch_size, seq_len, hidden_size]

        # 调用ITHP模型，传入epoch和max_epochs用于神经启发的模态排序
        b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, intermediate = self.ITHP(
            x, visual, acoustic, epoch=epoch, max_epochs=max_epochs
        )

        h_m = self.expand(b1)
        acoustic_vis_embedding = self.beta_shift * h_m

        sequence_output = self.dropout(self.LayerNorm(acoustic_vis_embedding + x))
        pooled_output = self.pooler(sequence_output)

        return pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1


class ITHP_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dberta = ITHP_DebertaModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, visual, acoustic, attention_mask=None, epoch=0, max_epochs=100):
        """
        前向传播
        
        参数:
            input_ids: 文本input ids
            visual: 视觉特征
            acoustic: 声学特征
            attention_mask: 注意力mask (可选)
            epoch: 当前训练轮次
            max_epochs: 最大训练轮次
        """
        pooled_output, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1 = self.dberta(
            input_ids, visual, acoustic, attention_mask=attention_mask, epoch=epoch, max_epochs=max_epochs
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1
