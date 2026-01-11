# src/code/model.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init

VERBOSE = False

def set_verbose(verbose_mode):
    """Set the global verbosity level"""
    global VERBOSE
    VERBOSE = verbose_mode

def encode_state(state, player_id=0):
    """
    Convert a Pokers state to a neural network input tensor.

    Args:
        state: The Pokers state
        player_id: The ID of the player for whom we're encoding
    Dimensions:
        Player hole cards (52 dimensions for card presence)
        Community cards (52 dimensions)
        Game stage (5 dimensions for preflop, flop, turn, river, showdown)
        Pot size (1 normalized)
        Button (N)
        Current player (N)
        Player states (Nx4: status, bet, chip, stake)
        Minimum bet (1)
        Legal actions (4)
        Previous actions (4+1)
    """
    encoded = []
    num_players = len(state.players_state)

    # Print debug info only if verbose
    if VERBOSE:
        print(f"Encoding state: current_player={state.current_player}, stage={state.stage}")
        print(f"Player states: {[(p.player, p.stake, p.bet_chips) for p in state.players_state]}")
        print(f"Pot: {state.pot}")

    # player's hole cards
    hand_cards = state.players_state[state.current_player].hand
    hand_enc = np.zeros(52)
    for card in hand_cards:
        card_idx = int(card.suit) * 13 + int(card.rank)
        hand_enc[card_idx] = 1
    encoded.append(hand_enc)

    # community cards
    community_enc = np.zeros(52)
    for card in state.public_cards:
        card_idx = int(card.suit) * 13 + int(card.rank)
        community_enc[card_idx] = 1
    encoded.append(community_enc)

    # game stage: Pre-flop, Flop, Turn, River, Showdown
    stage_enc = np.zeros(5)
    stage_enc[int(state.stage)] = 1
    encoded.append(stage_enc)

    # initial stake - prevent division by zero
    initial_stake = state.players_state[0].stake
    if initial_stake <= 0:
        if VERBOSE:
            print(f"WARNING: Initial stake is zero or negative: {initial_stake}")
        initial_stake = 1.0  # Use 1.0 as a fallback to prevent division by zero

    # pot size (normalized by initial stake)
    pot_enc = [state.pot / initial_stake]
    encoded.append(pot_enc)

    # button position
    button_enc = np.zeros(num_players)
    button_enc[state.button] = 1
    encoded.append(button_enc)

    # current player
    current_player_enc = np.zeros(num_players)
    current_player_enc[state.current_player] = 1
    encoded.append(current_player_enc)

    # player states
    for p in range(num_players):
        player_state = state.players_state[p]
        # Active status
        active_enc = 1.0 if player_state.active else 0.0
        # Current bet
        bet_enc = player_state.bet_chips / initial_stake
        # Pot chips (already won)
        pot_chips_enc = player_state.pot_chips / initial_stake
        # Remaining stake
        stake_enc = player_state.stake / initial_stake

        encoded.append(np.array([active_enc, bet_enc, pot_chips_enc, stake_enc]))

    # minimum bet
    min_bet_enc = [state.min_bet / initial_stake]
    encoded.append(min_bet_enc)

    # legal actions
    legal_actions_enc = np.zeros(4)  # Fold, Check, Call, Raise
    for action_enum in state.legal_actions:
        legal_actions_enc[int(action_enum)] = 1
    encoded.append(legal_actions_enc)

    # previous action if available
    prev_action_enc = np.zeros(4 + 1)  # Action type + normalized amount
    if state.from_action is not None:
        prev_action_enc[int(state.from_action.action.action)] = 1
        prev_action_enc[4] = state.from_action.action.amount / initial_stake
    encoded.append(prev_action_enc)

    # Concatenate all features
    return np.concatenate(encoded)


class PokerNetwork(nn.Module):
    """Poker network with continuous bet sizing capabilities (数值稳定版)."""

    def __init__(self, input_size=512, hidden_size=256, num_actions=3):
        super().__init__()
        # 1. 共享特征提取层：加入LayerNorm稳定中间层数值，避免梯度爆炸
        self.base = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 关键：层归一化，稳定ReLU输入分布
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )

        # 2. 动作类型预测头（fold/check/call/raise）
        self.action_head = nn.Linear(hidden_size, num_actions)

        # 3. 连续下注尺寸预测头：优化激活链，避免输出漂移
        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),  # 小层也加归一化，稳定Tanh输入
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出约束在0-1，后续缩放更可控
        )

        # 4. 关键：定制化参数初始化（从源头避免数值爆炸）
        self._init_weights()

    def _init_weights(self):
        """
        定制化初始化策略：针对不同激活函数适配初始化，避免初始输出/梯度爆炸
        - ReLU层：Kaiming Normal（适配ReLU的均值/方差）
        - Tanh/Sigmoid层：Xavier Normal（适配对称激活函数）
        - 输出层：小权重初始化，避免初始输出过大
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 判断所属层，定制初始化
                if any([isinstance(p, nn.ReLU) for p in m._forward_pre_hooks.values()]):
                    # ReLU前的Linear层：Kaiming Normal（修正ReLU的方差衰减）
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif any([isinstance(p, nn.Tanh) for p in m._forward_pre_hooks.values()]):
                    # Tanh前的Linear层：Xavier Normal（保持均值/方差稳定）
                    init.xavier_normal_(m.weight, gain=init.calculate_gain('tanh'))
                elif m in [self.action_head, self.sizing_head[-2]]:
                    # 输出层：小权重初始化（避免初始输出极端）
                    init.normal_(m.weight, mean=0.0, std=0.01)
                # 偏置项默认初始化为0（或小常数，避免偏移过大）
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass with numerical stability guarantees.

        Args:
            x: The state representation tensor (shape: [batch, input_size])

        Returns:
            Tuple of (action_logits, bet_size_prediction)
        """
        # 确保输入是float32（避免float16下溢/上溢）
        x = x.to(torch.float32)

        # 1. 基础特征提取（已做层归一化，数值稳定）
        features = self.base(x)

        # 2. 动作logits：加入clamp避免极端值（防御性措施）
        action_logits = self.action_head(features)
        action_logits = torch.clamp(action_logits, min=-10.0, max=10.0)  # 限制logits范围

        # 3. 下注尺寸：缩放后再clamp，确保在合理区间（0.1~3.0）
        bet_size_raw = self.sizing_head(features)
        bet_size = 0.1 + 2.9 * bet_size_raw
        bet_size = torch.clamp(bet_size, min=0.1, max=3.0)  # 防止极端值

        return action_logits, bet_size