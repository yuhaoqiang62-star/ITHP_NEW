import torch
import torch.nn as nn
import torch.nn.functional as F
import global_configs
import math
from typing import Dict, Tuple, List


class NeuroInspiredModalityOrdering(nn.Module):
    """
    受神经机制启发的模态排序模块
    基于神经科学中的竞争性学习和注意力机制
    """

    def __init__(self, modal_dims: Dict[str, int], hidden_dim: int = 256, num_modalities: int = 3):
        super(NeuroInspiredModalityOrdering, self).__init__()
        self.modal_dims = modal_dims
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        
        # 模态名称
        self.modality_names = ['text', 'acoustic', 'visual']

        # =============== 1. 信息量评估网络 (信息理论启发) ===============
        self.information_estimators = nn.ModuleDict({
            'text': self._build_information_estimator(modal_dims.get('text', 768)),
            'acoustic': self._build_information_estimator(modal_dims.get('acoustic', 128)),
            'visual': self._build_information_estimator(modal_dims.get('visual', 256))
        })

        # =============== 2. 互补性评估网络 (协同学习启发) ===============
        self.complementarity_estimators = nn.ModuleDict({
            'text_acoustic': self._build_complementarity_network(
                modal_dims.get('text', 768), modal_dims.get('acoustic', 128)
            ),
            'text_visual': self._build_complementarity_network(
                modal_dims.get('text', 768), modal_dims.get('visual', 256)
            ),
            'acoustic_visual': self._build_complementarity_network(
                modal_dims.get('acoustic', 128), modal_dims.get('visual', 256)
            )
        })

        # =============== 3. 神经竞争层 (竞争学习启发) ===============
        self.competitive_layer = nn.Sequential(
            nn.Linear(num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_modalities)
        )

        # =============== 4. 排序决策网络 (强化学习启发) ===============
        self.ordering_policy = nn.Sequential(
            nn.Linear(num_modalities * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, math.factorial(num_modalities))  # 所有排列可能性
        )

        # =============== 5. 温度参数（控制排序的确定性） ===============
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.temperature_scheduler = 0.99  # 训练过程中逐渐降低温度
        
        # =============== 6. 历史记忆单元 (记忆网络启发) ===============
        self.ordering_memory = nn.GRUCell(input_size=num_modalities,hidden_size=hidden_dim)
        self.memory_hidden = None

        # =============== 7. 可解释性：排序评分存储 ===============
        self.ordering_scores = {}
        self.permutations = self._generate_permutations(num_modalities)

    def _build_information_estimator(self, input_dim: int) -> nn.Module:
        """
        构建信息量评估网络
        评估单个模态的信息丰富程度 (基于熵)
        """
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def _build_complementarity_network(self, dim1: int, dim2: int) -> nn.Module:
        """
        构建互补性评估网络
        评估两个模态之间的互补程度 (基于相关性)
        """
        combined_dim = dim1 + dim2
        return nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, 1),
            nn.Sigmoid()
        )

    def _generate_permutations(self, n: int) -> List[Tuple]:
        """生成所有可能的模态排列"""
        from itertools import permutations
        indices = list(range(n))
        return list(permutations(indices))

    def estimate_modality_information(self, text: torch.Tensor, acoustic: torch.Tensor, 
                                     visual: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        估计每个模态的信息量
        使用信息论中的熵概念
        """
        # 平均池化以获得全局特征表示
        text_feat = text.mean(dim=1) if text.dim() > 2 else text
        acoustic_feat = acoustic.mean(dim=1) if acoustic.dim() > 2 else acoustic
        visual_feat = visual.mean(dim=1) if visual.dim() > 2 else visual

        info_scores = {
            'text': self.information_estimators['text'](text_feat),
            'acoustic': self.information_estimators['acoustic'](acoustic_feat),
            'visual': self.information_estimators['visual'](visual_feat)
        }
        return info_scores

    def estimate_complementarity(self, text: torch.Tensor, acoustic: torch.Tensor,
                                visual: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        估计模态之间的互补性
        基于多模态特征的相关性和独特性
        """
        text_feat = text.mean(dim=1) if text.dim() > 2 else text
        acoustic_feat = acoustic.mean(dim=1) if acoustic.dim() > 2 else acoustic
        visual_feat = visual.mean(dim=1) if visual.dim() > 2 else visual

        complementarity = {
            'text_acoustic': self.complementarity_estimators['text_acoustic'](
                torch.cat([text_feat, acoustic_feat], dim=-1)
            ),
            'text_visual': self.complementarity_estimators['text_visual'](
                torch.cat([text_feat, visual_feat], dim=-1)
            ),
            'acoustic_visual': self.complementarity_estimators['acoustic_visual'](
                torch.cat([acoustic_feat, visual_feat], dim=-1)
            )
        }
        return complementarity

    def compute_ordering_scores(self, info_scores: Dict[str, torch.Tensor],
                               complementarity: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算每种排列方式的评分
        使用竞争学习机制选择最优排序
        
        排序评分基于：
        1. 模态信息量 (主模态应该信息丰富)
        2. 互补性 (后续模态应该互补前面的)
        3. 信息流效率 (避免冗余)
        """
        batch_size = info_scores['text'].shape[0]
        num_permutations = len(self.permutations)

        # 初始化排列评分 [batch_size, num_permutations]
        perm_scores = torch.zeros(batch_size, num_permutations, device=info_scores['text'].device)

        info_values = torch.stack([
            info_scores['text'].squeeze(-1),
            info_scores['acoustic'].squeeze(-1),
            info_scores['visual'].squeeze(-1)
        ], dim=1)  # [batch_size, 3]

        for perm_idx, perm in enumerate(self.permutations):
            # 对于每个排列计算评分
            score = torch.zeros_like(info_values[:, 0])

            # 1. 主模态信息量得分 (第一个模态应该有最多信息)
            primary_idx = perm[0]
            score += info_values[:, primary_idx] * 0.5

            # 2. 互补性得分 (后续模态的互补性)
            if len(perm) > 1:
                for i in range(len(perm) - 1):
                    curr_idx = perm[i]
                    next_idx = perm[i + 1]
                    
                    # 获取对应的互补性评分
                    comp_key = f"{self.modality_names[curr_idx]}_{self.modality_names[next_idx]}"
                    if comp_key in complementarity:
                        comp_score = complementarity[comp_key].squeeze(-1)
                        # 互补性高的排序得分更高
                        score += comp_score * (0.25 / (i + 1))

            perm_scores[:, perm_idx] = score

        return perm_scores

    def select_ordering(self, perm_scores: torch.Tensor, use_gumbel_softmax: bool = True) -> Tuple:
        """
        使用温度参数和Gumbel-Softmax选择排列
        这模拟了神经元竞争性激活的过程
        
        返回：选择的排列索引和对应的软权重
        """
        batch_size = perm_scores.shape[0]

        if use_gumbel_softmax and self.training:
            # 使用Gumbel-Softmax进行可微的排列选择
            # 这允许梯度流向排列选择过程
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(perm_scores) + 1e-8) + 1e-8)
            logits = (perm_scores + gumbel_noise) / self.temperature
            soft_weights = F.softmax(logits, dim=-1)
            
            # 获取最高评分的排列
            hard_selection = torch.argmax(soft_weights, dim=1)
        else:
            # 推理时使用确定性选择
            soft_weights = F.softmax(perm_scores / self.temperature, dim=-1)
            hard_selection = torch.argmax(soft_weights, dim=1)

        # 将选择转换为实际排列
        selected_permutations = [self.permutations[idx] for idx in hard_selection.cpu().numpy()]

        return hard_selection, soft_weights, selected_permutations

    def reorder_modalities(self, modalities: Dict[str, torch.Tensor],
                          permutation: Tuple[int, int, int]) -> Tuple[Dict, List]:
        """
        根据排列重新组织模态
        返回重新排序的模态字典和处理顺序信息
        """
        names = ['text', 'acoustic', 'visual']
        ordered_modalities = {}
        processing_order = []

        for position, modal_idx in enumerate(permutation):
            modal_name = names[modal_idx]
            ordered_modalities[f'modal_{position}'] = modalities[modal_name]
            processing_order.append(modal_name)

        return ordered_modalities, processing_order

    def update_memory(self, ordering_indices: torch.Tensor):
        """
    更新神经记忆，记住过去选择的排序
    使用GRU Cell实现
    
    Args:
        ordering_indices: shape [batch_size]，每个样本选择的排列索引
    Returns:
        memory_hidden: shape [batch_size, hidden_dim]
         """
        batch_size = ordering_indices.shape[0]
        device = ordering_indices.device
    
    # 初始化或重置隐藏状态（如果batch_size变化）
        if self.memory_hidden is None or self.memory_hidden.shape[0] != batch_size:
            self.memory_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)

    # 将排列索引归一化到 [0, 1]
    # ordering_indices shape: [batch_size]
        normalized_idx = ordering_indices.float() / len(self.permutations)
    
    # 扩展为 [batch_size, num_modalities] 的输入
    # GRUCell 需要输入 shape: [batch_size, input_size]
        memory_input = normalized_idx.unsqueeze(1).repeat(1, self.num_modalities)
    
    # 确保维度正确
        assert memory_input.shape == (batch_size, self.num_modalities), \
        f"Expected memory_input shape ({batch_size}, {self.num_modalities}), got {memory_input.shape}"
    
    # 更新记忆状态
        self.memory_hidden = self.ordering_memory(memory_input, self.memory_hidden)


        return self.memory_hidden

    def decay_temperature(self):
        """衰减温度参数，使排列选择逐渐变得确定"""
        with torch.no_grad():
            self.temperature.data *= self.temperature_scheduler

    def forward(self, text: torch.Tensor, acoustic: torch.Tensor, 
               visual: torch.Tensor, epoch: int = 0, max_epochs: int = 100) -> Dict:
        """
        完整的神经启发排序流程
        
        返回：
        {
            'ordered_modalities': 重新排序的模态字典,
            'processing_order': 处理顺序列表,
            'selected_permutation': 选择的排列,
            'ordering_confidence': 排列选择的置信度,
            'info_scores': 各模态的信息量评分,
            'complementarity': 互补性评分,
            'perm_scores': 所有排列的评分
        }
        """
        # 步骤1：评估每个模态的信息量
        info_scores = self.estimate_modality_information(text, acoustic, visual)

        # 步骤2：评估模态间的互补性
        complementarity = self.estimate_complementarity(text, acoustic, visual)

        # 步骤3：使用竞争学习计算排列评分
        perm_scores = self.compute_ordering_scores(info_scores, complementarity)

        # 步骤4：选择最优排列（软选择和硬选择）
        ordering_indices, soft_weights, selected_perms = self.select_ordering(perm_scores)

        # 步骤5：更新神经记忆
        memory_state = self.update_memory(ordering_indices)

        # 步骤6：重新排序模态
        modalities_dict = {'text': text, 'acoustic': acoustic, 'visual': visual}
        ordered_modalities_list = []
        processing_orders = []

        for i, perm in enumerate(selected_perms):
            ordered, order = self.reorder_modalities(modalities_dict, perm)
            ordered_modalities_list.append(ordered)
            processing_orders.append(order)

        # 步骤7：动态调整温度（模拟神经适应过程）
        if self.training:
            progress = epoch / max_epochs
            # 线性衰减温度
            new_temp = 1.0 * (1.0 - 0.5 * progress)
            with torch.no_grad():
                self.temperature.data.fill_(new_temp)

        # 计算排列选择的置信度
        max_scores = torch.max(soft_weights, dim=1)[0]
        ordering_confidence = max_scores

        # 存储排序信息用于调试和可视化
        self.ordering_scores = {
            'perm_scores': perm_scores.detach().cpu(),
            'soft_weights': soft_weights.detach().cpu(),
            'selected_indices': ordering_indices.detach().cpu(),
            'confidence': ordering_confidence.detach().cpu()
        }

        return {
            'ordered_modalities': ordered_modalities_list,
            'processing_order': processing_orders,
            'selected_permutation': selected_perms,
            'ordering_confidence': ordering_confidence,
            'info_scores': {k: v.detach() for k, v in info_scores.items()},
            'complementarity': {k: v.detach() for k, v in complementarity.items()},
            'perm_scores': perm_scores.detach(),
            'soft_weights': soft_weights.detach(),
            'memory_state': memory_state.detach()
        }


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim=256, num_heads=8):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, query_dim)

        self.layer_norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(0.1)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.shape[:2]

        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attended_values = torch.matmul(attention_weights, V)

        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )

        output = self.output_proj(attended_values)
        output = self.layer_norm(output + query)

        return output, attention_weights.mean(dim=1)


class LearnableWeights(nn.Module):
    """可学习的损失权重参数"""

    def __init__(self, initial_beta=8.0, initial_gamma=32.0, initial_lambda=0.3):
        super(LearnableWeights, self).__init__()

        self.log_beta = nn.Parameter(torch.log(torch.tensor(initial_beta, dtype=torch.float32)))
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(initial_gamma, dtype=torch.float32)))
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(initial_lambda, dtype=torch.float32)))

        self.temperature = nn.Parameter(torch.tensor(1.0))

    @property
    def beta(self):
        return torch.exp(self.log_beta / self.temperature)

    @property
    def gamma(self):
        return torch.exp(self.log_gamma / self.temperature)

    @property
    def lambda_weight(self):
        return torch.exp(self.log_lambda / self.temperature)

    def get_weights(self):
        return {
            'beta': self.beta.item(),
            'gamma': self.gamma.item(),
            'lambda': self.lambda_weight.item(),
            'temperature': self.temperature.item()
        }


class GatedBottleneck(nn.Module):
    """门控瓶颈机制"""

    def __init__(self, input_dim, bottleneck_dim, gate_activation='sigmoid'):
        super(GatedBottleneck, self).__init__()

        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.Sigmoid() if gate_activation == 'sigmoid' else nn.Tanh()
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
        )

        self.importance_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        importance_weights = self.importance_gate(x)
        gated_input = x * importance_weights
        gated_input = self.layer_norm(gated_input + x)

        gate_values = self.gate_network(gated_input)

        h = self.encoder(gated_input)
        mu, logvar = h.chunk(2, dim=-1)

        gated_mu = mu * gate_values

        gate_logvar_effect = torch.log(gate_values + 1e-8) * 0.1
        gated_logvar = logvar + gate_logvar_effect

        return gated_mu, gated_logvar, gate_values, importance_weights


class ITHP(nn.Module):
    """
    改进的ITHP模型，集成神经启发的模态排序机制
    """

    def __init__(self, ITHP_args):
        super(ITHP, self).__init__()

        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (global_configs.TEXT_DIM, global_configs.ACOUSTIC_DIM,
                                              global_configs.VISUAL_DIM)

        self.X0_dim = ITHP_args['X0_dim']
        self.X1_dim = ITHP_args['X1_dim']
        self.X2_dim = ITHP_args['X2_dim']
        self.inter_dim = ITHP_args['inter_dim']
        self.drop_prob = ITHP_args['drop_prob']
        self.max_sen_len = ITHP_args['max_sen_len']
        self.B0_dim = ITHP_args['B0_dim']
        self.B1_dim = ITHP_args['B1_dim']

        # ===== 神经启发模态排序模块 =====
        modal_dims = {
            'text': self.X0_dim,
            'acoustic': self.X1_dim,
            'visual': self.X2_dim
        }
        self.neuro_ordering = NeuroInspiredModalityOrdering(
            modal_dims=modal_dims,
            hidden_dim=256,
            num_modalities=3
        )

        # 可学习权重
        self.learnable_weights = LearnableWeights(
            initial_beta=ITHP_args.get('p_beta', 8.0),
            initial_gamma=ITHP_args.get('p_gamma', 32.0),
            initial_lambda=ITHP_args.get('p_lambda', 0.3)
        )

        # 第一层门控瓶颈编码器
        self.gated_encoder1 = GatedBottleneck(
            self.X0_dim, self.B0_dim, gate_activation='sigmoid'
        )

        # 跨模态注意力
        self.text_acoustic_attention = CrossModalAttention(
            query_dim=self.B0_dim,
            key_dim=self.X1_dim,
            value_dim=self.X1_dim,
            hidden_dim=min(256, self.B0_dim),
            num_heads=8
        )

        # 第一层MLP
        self.MLP1 = nn.Sequential(
            nn.Linear(self.B0_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X1_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        # 第二层门控瓶颈编码器
        self.gated_encoder2 = GatedBottleneck(
            self.B0_dim, self.B1_dim, gate_activation='sigmoid'
        )

        # 跨模态注意力
        self.b0_visual_attention = CrossModalAttention(
            query_dim=self.B1_dim,
            key_dim=self.X2_dim,
            value_dim=self.X2_dim,
            hidden_dim=min(256, self.B1_dim),
            num_heads=8
        )

        # 第二层MLP
        self.MLP2 = nn.Sequential(
            nn.Linear(self.B1_dim, self.inter_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.inter_dim, self.X2_dim),
            nn.Sigmoid(),
            nn.Dropout(self.drop_prob),
        )

        # 多模态融合层
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(self.B1_dim + self.X1_dim + self.X2_dim, self.B1_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.B1_dim, self.B1_dim),
        )

        self.criterion = nn.MSELoss()
        self.attention_weights = {}

    def kl_loss(self, mu, logvar):
        """计算KL散度损失"""
        kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_mean = torch.mean(kl_div)
        return kl_mean

    def reparameterise(self, mu, logvar):
        """重参数化技巧"""
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def adaptive_weight_decay(self, epoch, max_epochs):
        """自适应权重衰减"""
        progress = epoch / max_epochs
        decay_factor = 1.0 - 0.5 * progress
        return decay_factor

    def forward(self, x, visual, acoustic, epoch=0, max_epochs=100):
        """
        前向传播，集成神经启发的模态排序
        
        x: 文本特征 [batch_size, seq_len, text_dim]
        acoustic: 声学特征 [batch_size, seq_len, acoustic_dim]
        visual: 视觉特征 [batch_size, seq_len, visual_dim]
        """
        
        # ===== 步骤1：神经启发模态排序 =====
        ordering_result = self.neuro_ordering(x, acoustic, visual, epoch, max_epochs)
        
        processing_order = ordering_result['processing_order']
        ordering_confidence = ordering_result['ordering_confidence']

        # 获取当前可学习权重
        current_weights = self.learnable_weights.get_weights()
        beta = self.learnable_weights.beta
        gamma = self.learnable_weights.gamma
        lambda_weight = self.learnable_weights.lambda_weight

        decay_factor = self.adaptive_weight_decay(epoch, max_epochs)

        # ===== 步骤2：根据学习到的顺序处理模态 =====
        # 为简化起见，这里使用标准的处理流程
        # 但实际应用中可以根据 processing_order 动态调整处理流程
        
        # 第一层处理
        mu1, logvar1, gate1_values, importance1_weights = self.gated_encoder1(x)
        kl_loss_0 = self.kl_loss(mu1, logvar1)

        b0 = self.reparameterise(mu1, logvar1)

        attended_b0, attention_weights_1 = self.text_acoustic_attention(
            query=b0, key=acoustic, value=acoustic
        )
        self.attention_weights['text_acoustic'] = attention_weights_1

        enhanced_b0 = b0 + 0.3 * attended_b0

        output1 = self.MLP1(enhanced_b0)
        mse_0 = self.criterion(output1, acoustic)

        IB0 = kl_loss_0 + beta * mse_0 * decay_factor

        # 第二层处理
        mu2, logvar2, gate2_values, importance2_weights = self.gated_encoder2(enhanced_b0)
        kl_loss_1 = self.kl_loss(mu2, logvar2)

        b1 = self.reparameterise(mu2, logvar2)

        attended_b1, attention_weights_2 = self.b0_visual_attention(
            query=b1, key=visual, value=visual
        )
        self.attention_weights['b0_visual'] = attention_weights_2

        enhanced_b1 = b1 + 0.3 * attended_b1

        output2 = self.MLP2(enhanced_b1)
        mse_1 = self.criterion(output2, visual)

        IB1 = kl_loss_1 + gamma * mse_1 * decay_factor

        # 多模态融合
        fusion_input = torch.cat([enhanced_b1, output1, output2], dim=-1)
        final_b1 = self.multimodal_fusion(fusion_input)

        # ===== 步骤3：加入排序置信度信息 =====
        # 使用排序置信度作为附加的损失项权重
        # 这鼓励模型学习有意义的模态排序
        ordering_consistency_loss = -torch.log(ordering_confidence.mean() + 1e-8)

        IB_total = IB0 + lambda_weight * IB1 + 0.01 * ordering_consistency_loss

        intermediate_results = {
            'gate1_values': gate1_values,
            'gate2_values': gate2_values,
            'importance1_weights': importance1_weights,
            'importance2_weights': importance2_weights,
            'attention_weights': self.attention_weights.copy(),
            'current_weights': current_weights,
            'enhanced_b0': enhanced_b0,
            'enhanced_b1': enhanced_b1,
            'reconstructed_acoustic': output1,
            'reconstructed_visual': output2,
            'ordering_info': {
                'processing_order': processing_order,
                'ordering_confidence': ordering_confidence.detach(),
                'info_scores': ordering_result['info_scores'],
                'complementarity': ordering_result['complementarity'],
                'perm_scores': ordering_result['perm_scores']
            }
        }

        return final_b1, IB_total, kl_loss_0, mse_0, kl_loss_1, mse_1, intermediate_results


# ================== 改进的训练循环 ==================

class ImprovedTrainingLoopWithOrdering:
    """集成神经启发模态排序的改进训练循环"""

    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.training_stats = {
            'weights_history': [],
            'attention_stats': [],
            'gate_stats': [],
            'ordering_stats': []
        }

    def train_epoch_improved(self, train_dataloader, epoch, max_epochs):
        """改进的训练epoch，包含模态排序监控"""
        self.model.train()
        epoch_stats = {
            'total_loss': 0.0,
            'ib_loss': 0.0,
            'kl_losses': [],
            'mse_losses': [],
            'weight_changes': {},
            'attention_entropy': [],
            'ordering_confidence': [],
            'modality_orders': {'text_acoustic_visual': 0, 'text_visual_acoustic': 0,
                               'acoustic_text_visual': 0, 'acoustic_visual_text': 0,
                               'visual_text_acoustic': 0, 'visual_acoustic_text': 0}
        }

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, visual, acoustic, label_ids = batch

            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            # 前向传播
            logits, IB_loss, kl_loss_0, mse_0, kl_loss_1, mse_1, intermediate = self.model(
                input_ids, visual_norm, acoustic_norm, epoch, max_epochs
            )

            # 主任务损失
            loss_fct = nn.MSELoss()
            main_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            # 总损失
            total_loss = main_loss + 0.1 * IB_loss

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            # 统计收集
            epoch_stats['total_loss'] += total_loss.item()
            epoch_stats['ib_loss'] += IB_loss.item()
            epoch_stats['kl_losses'].append([kl_loss_0.item(), kl_loss_1.item()])
            epoch_stats['mse_losses'].append([mse_0.item(), mse_1.item()])

            current_weights = intermediate['current_weights']
            epoch_stats['weight_changes'] = current_weights

            # 收集排序相关统计
            ordering_info = intermediate['ordering_info']
            ordering_conf = ordering_info['ordering_confidence'].mean().item()
            epoch_stats['ordering_confidence'].append(ordering_conf)

            # 统计各种排序方式出现的次数
            for order in ordering_info['processing_order']:
                order_key = '_'.join(order)
                if order_key in epoch_stats['modality_orders']:
                    epoch_stats['modality_orders'][order_key] += 1

            # 计算注意力熵
            for key, attention_weights in intermediate['attention_weights'].items():
                entropy = self.calculate_attention_entropy(attention_weights)
                epoch_stats['attention_entropy'].append(entropy.item())

        # 计算平均统计
        num_batches = len(train_dataloader)
        epoch_stats['total_loss'] /= num_batches
        epoch_stats['ib_loss'] /= num_batches
        epoch_stats['avg_attention_entropy'] = (
            sum(epoch_stats['attention_entropy']) / len(epoch_stats['attention_entropy'])
            if epoch_stats['attention_entropy'] else 0
        )
        epoch_stats['avg_ordering_confidence'] = (
            sum(epoch_stats['ordering_confidence']) / len(epoch_stats['ordering_confidence'])
            if epoch_stats['ordering_confidence'] else 0
        )

        self.training_stats['weights_history'].append(epoch_stats['weight_changes'])
        self.training_stats['attention_stats'].append(epoch_stats['avg_attention_entropy'])
        self.training_stats['ordering_stats'].append({
            'confidence': epoch_stats['avg_ordering_confidence'],
            'orders': epoch_stats['modality_orders'].copy()
        })

        return epoch_stats

    def calculate_attention_entropy(self, attention_weights):
        """计算注意力权重的熵"""
        batch_size = attention_weights.shape[0]
        entropies = []

        for i in range(batch_size):
            attn = attention_weights[i].mean(dim=0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            entropy = -torch.sum(attn * torch.log(attn + 1e-8), dim=-1).mean()
            entropies.append(entropy)

        return torch.stack(entropies).mean()

    def print_training_stats(self, epoch, epoch_stats):
        """打印训练统计信息，包括模态排序信息"""
        print(f"\n=== Epoch {epoch + 1} Stats ===")
        print(f"Total Loss: {epoch_stats['total_loss']:.4f}")
        print(f"IB Loss: {epoch_stats['ib_loss']:.4f}")
        print(f"Current Weights - Beta: {epoch_stats['weight_changes']['beta']:.2f}, "
              f"Gamma: {epoch_stats['weight_changes']['gamma']:.2f}, "
              f"Lambda: {epoch_stats['weight_changes']['lambda']:.2f}")
        print(f"Average Attention Entropy: {epoch_stats['avg_attention_entropy']:.4f}")
        print(f"Average Ordering Confidence: {epoch_stats['avg_ordering_confidence']:.4f}")
        print(f"KL Losses: Layer1={epoch_stats['kl_losses'][-1][0]:.4f}, "
              f"Layer2={epoch_stats['kl_losses'][-1][1]:.4f}")
        print(f"MSE Losses: Acoustic={epoch_stats['mse_losses'][-1][0]:.4f}, "
              f"Visual={epoch_stats['mse_losses'][-1][1]:.4f}")
        print(f"\nModality Ordering Distribution:")
        for order, count in epoch_stats['modality_orders'].items():
            percentage = (count / sum(epoch_stats['modality_orders'].values())) * 100
            print(f"  {order}: {count} ({percentage:.1f}%)")


# ================== 使用示例 ==================

def create_ithp_model_with_ordering(original_args):
    """创建集成神经启发模态排序的ITHP模型"""
    
    improved_args = original_args.copy()
    improved_args.update({
        'use_neuro_ordering': True,
        'use_attention': True,
        'use_gating': True,
        'use_learnable_weights': True,
        'attention_heads': 8,
    })

    return ITHP(improved_args)


if __name__ == "__main__":
    """
    使用示例代码
    """
    ITHP_args = {
        'X0_dim': 768,  # TEXT_DIM (DeBERTa hidden size)
        'X1_dim': 128,  # ACOUSTIC_DIM
        'X2_dim': 256,  # VISUAL_DIM
        'inter_dim': 256,
        'drop_prob': 0.1,
        'max_sen_len': 50,
        'B0_dim': 128,
        'B1_dim': 128,
        'p_beta': 1.0,
        'p_gamma': 1.0,
        'p_lambda': 1.0,
    }

    # 创建模型
    model = ITHP(ITHP_args)
    print("ITHP Model with Neuro-Inspired Modality Ordering created successfully!")
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 打印神经启发模态排序模块的信息
    print(f"\nNeuro-Inspired Modality Ordering Module:")
    print(f"  - Number of possible permutations: {len(model.neuro_ordering.permutations)}")
    print(f"  - Permutations: {model.neuro_ordering.permutations}")
    print(f"  - Initial temperature: {model.neuro_ordering.temperature.item():.4f}")
