# deep_cfr.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pokers
from collections import deque

from core.memory import PrioritizedMemory
from src.core.model import PokerNetwork, encode_state, VERBOSE
from src.utils.settings import STRICT_CHECKING
from src.utils.logging import log_game_error
from src.utils.state_control import get_legal_action_types, action_type_to_pokers_action


class SelfCFR:
    def __init__(self, num_players=2, memory_size=300000, device='cpu'):
        self.num_players = num_players
        self.device = device
        
        # Define action types (Fold, Check/Call, Raise)
        self.num_actions = 3
        
        # Calculate input size based on state encoding
        input_size = 52 + 52 + 5 + 1 + num_players + num_players + num_players*4 + 1 + 4 + 5
        
        # Regret network
        self.regret_net = PokerNetwork(input_size=input_size, hidden_size=256, num_actions=self.num_actions).to(device)
        self.regret_optimizer = optim.Adam(self.regret_net.parameters(), lr=1e-6, weight_decay=1e-5)
        self.regret_memory = PrioritizedMemory(memory_size)
        
        # Strategy network
        self.strategy_net = PokerNetwork(input_size=input_size, hidden_size=256, num_actions=self.num_actions).to(device)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.00005, weight_decay=1e-5)
        self.strategy_memory = deque(maxlen=memory_size)
        
        # For keeping statistics
        self.iteration = 0
        # Regret normalization tracker
        self.max_regret_seen = 1.0

    def cfr_traverse(self, state, player_id, depth=0):
        """
        Traverse the game tree using external sampling MonteCarloCFR with continuous bet sizing.
        
        Args:
            state: Current game state
            depth: Current recursion depth
            
        Returns:
            Expected value for the current player
        """
        # Add recursion depth protection
        max_depth = 1000
        if depth > max_depth:
            if VERBOSE:
                print(f"WARNING: Max recursion depth reached ({max_depth}). Returning zero value.")
            return 0

        if state.final_state:
            # Return payoff for the trained agent
            return state.players_state[0].reward
        # encode the state
        current_player = state.current_player
        state_encoded = encode_state(state)

        # If it's the trained agent's turn
        if current_player != player_id:
            # Let the random agent choose an action
            action = self.choose_action(state)
            new_state = state.apply_action(action)

            # Check if the action was valid
            if new_state.status != pokers.StateStatus.Ok:
                log_file = log_game_error(state, action, f"State status not OK ({new_state.status})")
                if STRICT_CHECKING:
                    raise ValueError(
                        f"State status not OK ({new_state.status}) from random agent. Details logged to {log_file}")
                if VERBOSE:
                    print(f"WARNING: Random agent made invalid action at depth {depth}. Status: {new_state.status}")
                    print(f"Details logged to {log_file}")
                return 0

            return self.cfr_traverse(new_state, player_id, depth + 1)
        else:
            legal_action_types = get_legal_action_types(state)
            
            if not legal_action_types:
                if VERBOSE:
                    print(f"WARNING: No legal actions found for player {current_player} at depth {depth}")
                return 0
                
            # Encode the base state
            state_tensor = torch.FloatTensor(state_encoded).to(self.device)
            
            # Get regrets and bet sizing prediction from network
            with torch.no_grad():
                regrets, bet_predicts = self.regret_net(state_tensor.unsqueeze(0))
                regrets = regrets[0].cpu().numpy()
                bet_multiplier = bet_predicts[0][0].item()
            
            # Use regret matching to compute strategy for action types
            regrets_masked = np.zeros(self.num_actions)
            for a in legal_action_types:
                regrets_masked[a] = max(regrets[a], 0)
                
            # Choose an action based on the strategy
            if sum(regrets_masked) > 0:
                strategy = regrets_masked / sum(regrets_masked)
            else:
                strategy = np.zeros(self.num_actions)
                for a in legal_action_types:
                    strategy[a] = 1.0 / len(legal_action_types)
            
            # Choose actions and traverse
            action_values = np.zeros(self.num_actions)
            for action_type in legal_action_types:
                try:
                    # Use the predicted bet size for raise actions
                    if action_type == 2:  # Raise
                        pokers_action = action_type_to_pokers_action(action_type, state, bet_multiplier)
                    else:
                        pokers_action = action_type_to_pokers_action(action_type, state)
                    
                    new_state = state.apply_action(pokers_action)
                    
                    # Check if the action was valid
                    if new_state.status != pokers.StateStatus.Ok:
                        log_file = log_game_error(state, pokers_action, f"State status not OK ({new_state.status})")
                        if STRICT_CHECKING:
                            raise ValueError(f"State status not OK ({new_state.status}) during CFR traversal. Details logged to {log_file}")
                        elif VERBOSE:
                            print(f"WARNING: Invalid action {action_type} at depth {depth}. Status: {new_state.status}")
                            print(f"Player: {current_player}, Action: {pokers_action.action}, Amount: {pokers_action.amount if pokers_action.action == pokers.ActionEnum.Raise else 'N/A'}")
                            print(f"Current bet: {state.players_state[current_player].bet_chips}, Stake: {state.players_state[current_player].stake}")
                            print(f"Details logged to {log_file}")
                        continue  # Skip this action and try others in non-strict mode
                        
                    action_values[action_type] = self.cfr_traverse(new_state, player_id, depth + 1)
                except Exception as e:
                    if VERBOSE:
                        print(f"ERROR in traversal for action {action_type}: {e}")
                    action_values[action_type] = 0
                    if STRICT_CHECKING:
                        raise  # Re-raise in strict mode
            
            # Compute counterfactual regrets and add to memory
            ev = sum(strategy[a] * action_values[a] for a in legal_action_types)
            
            # Calculate normalization factor
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            
            for action_type in legal_action_types:
                # Calculate regret
                regret = action_values[action_type] - ev
                
                # Normalize and clip regret
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -5.0, 5.0)

                # Store in prioritized memory with regret magnitude as priority
                priority = abs(clipped_regret) + 0.01  # Add small constant to ensure non-zero priority

                # For raise actions, store the bet size multiplier
                self.regret_memory.add(
                    (state_encoded,
                     action_type,
                     bet_multiplier if action_type == 2 else 0.0,
                     clipped_regret),
                     priority
                )
            # Add to strategy memory
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_types:
                strategy_full[a] = strategy[a]
            
            self.strategy_memory.append((
                state_encoded,
                strategy_full,
                bet_multiplier if 2 in legal_action_types else 0.0
            ))
            
            return ev

    def train_regret_net(self, batch_size=128, epochs=3, beta_start=0.4, beta_end=1.0):
        """
        Train the regret network using prioritized experience replay.
        """
        if len(self.regret_memory) < batch_size:
            return 0
        
        self.regret_net.train()
        total_loss = 0
        
        # Calculate current beta for importance sampling
        progress = min(1.0, self.iteration / 10000)
        beta = beta_start + progress * (beta_end - beta_start)
        
        for epoch in range(epochs):
            # Sample batch from prioritized memory with current beta
            batch, indices, weights = self.regret_memory.sample(batch_size, beta=beta)
            states, action_types, bet_sizes, regrets = zip(*batch)
            
            # [DEBUG 1] Log regret values in memory
            if self.iteration % 100 == 0 and epoch == 0:
                regret_array = np.array(regrets)
                print(f"[DEBUG-MEMORY] Regret stats: min={np.min(regret_array):.2f}, max={np.max(regret_array):.2f}, mean={np.mean(regret_array):.2f}")
            
            # Convert to tensors
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            action_type_tensors = torch.LongTensor(np.array(action_types)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
            weight_tensors = torch.FloatTensor(weights).to(self.device)
            
            # Forward pass
            action_advantages, bet_size_predicts = self.regret_net(state_tensors)
            
            # [DEBUG 2] Log network raw outputs to identify explosion
            if self.iteration % 100 == 0 and epoch == 0:
                with torch.no_grad():
                    max_adv = torch.max(action_advantages).item()
                    min_adv = torch.min(action_advantages).item()
                    print(f"[DEBUG-NETWORK] Network outputs: min={min_adv:.2f}, max={max_adv:.2f}")
            
            # Compute action type loss (for all actions)
            predicted_regrets = action_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
            
            # [DEBUG 3] Log gathered predictions
            if self.iteration % 100 == 0 and epoch == 0:
                with torch.no_grad():
                    max_predict = torch.max(predicted_regrets).item()
                    min_predict = torch.min(predicted_regrets).item()
                    max_target = torch.max(regret_tensors).item()
                    min_target = torch.min(regret_tensors).item()
                    print(f"[DEBUG-PRED] Predictions: min={min_predict:.2f}, max={max_predict:.2f}")
                    print(f"[DEBUG-TARGET] Targets: min={min_target:.2f}, max={max_target:.2f}")
            
            action_loss = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
            
            # [DEBUG 4] Log raw loss values before weighting
            if self.iteration % 100 == 0 and epoch == 0:
                with torch.no_grad():
                    max_loss = torch.max(action_loss).item()
                    mean_loss = torch.mean(action_loss).item()
                    print(f"[DEBUG-LOSS] Raw loss values: max={max_loss:.2f}, mean={mean_loss:.2f}")
            
            weighted_action_loss = (action_loss * weight_tensors).mean()
            
            # [DEBUG 5] Log weighted loss
            if self.iteration % 100 == 0 and epoch == 0:
                print(f"[DEBUG-WEIGHTED] Weighted action loss: {weighted_action_loss.item():.2f}")
                
                # Check for weight outliers
                max_weight = torch.max(weight_tensors).item()
                min_weight = torch.min(weight_tensors).item()
                print(f"[DEBUG-WEIGHTS] Weight range: min={min_weight:.4f}, max={max_weight:.4f}")
            
            # Compute bet sizing loss (only for raise actions)
            raise_mask = (action_type_tensors == 2)
            if torch.any(raise_mask):
                # Calculate loss for all bet sizes
                all_bet_losses = F.smooth_l1_loss(bet_size_predicts, bet_size_tensors, reduction='none')
                
                # Only count losses for raise actions, zero out others
                masked_bet_losses = all_bet_losses * raise_mask.float().unsqueeze(1)
                
                # Calculate weighted average loss
                raise_count = raise_mask.sum().item()
                if raise_count > 0:
                    weighted_bet_size_loss = (masked_bet_losses.squeeze() * weight_tensors).sum() / raise_count
                    combined_loss = weighted_action_loss + 0.5 * weighted_bet_size_loss
                    
                    # [DEBUG 6] Log bet size loss
                    if self.iteration % 100 == 0 and epoch == 0:
                        print(f"[DEBUG-BET] Weighted bet size loss: {weighted_bet_size_loss.item():.2f}")
                else:
                    combined_loss = weighted_action_loss
            else:
                combined_loss = weighted_action_loss
            
            # [DEBUG 7] Log final combined loss
            if self.iteration % 100 == 0 and epoch == 0:
                print(f"[DEBUG-COMBINED] Combined loss before clipping: {combined_loss.item():.2f}")
            
            # Backward pass and optimize
            self.regret_optimizer.zero_grad()
            combined_loss.backward()
            
            # [DEBUG 8] Check for gradient explosion before clipping
            if self.iteration % 100 == 0 and epoch == 0:
                total_grad_norm = 0
                max_layer_norm = 0
                max_layer_name = ""
                for name, param in self.regret_net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm * grad_norm
                        if grad_norm > max_layer_norm:
                            max_layer_norm = grad_norm
                            max_layer_name = name
                
                total_grad_norm = np.sqrt(total_grad_norm)
                print(f"[DEBUG-GRAD] Before clipping - Total grad norm: {total_grad_norm:.2f}")
                print(f"[DEBUG-GRAD] Largest layer grad: {max_layer_name} = {max_layer_norm:.2f}")
            
            # Apply gradient clipping (your existing code)
            torch.nn.utils.clip_grad_norm_(self.regret_net.parameters(), max_norm=0.5)
            
            # [DEBUG 9] Check effect of gradient clipping
            if self.iteration % 100 == 0 and epoch == 0:
                total_grad_norm = 0
                for name, param in self.regret_net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        total_grad_norm += grad_norm * grad_norm
                
                total_grad_norm = np.sqrt(total_grad_norm)
                print(f"[DEBUG-GRAD] After clipping - Total grad norm: {total_grad_norm:.2f}")
            
            self.regret_optimizer.step()
            
            # [DEBUG 10] Check for extreme parameter values after update
            if self.iteration % 1000 == 0 and epoch == 0:
                with torch.no_grad():
                    max_param_val = -float('inf')
                    max_param_name = ""
                    for name, param in self.regret_net.named_parameters():
                        param_max = torch.max(torch.abs(param)).item()
                        if param_max > max_param_val:
                            max_param_val = param_max
                            max_param_name = name
                    
                    print(f"[DEBUG-PARAMS] Largest parameter value: {max_param_name} = {max_param_val:.2f}")
            
            # Update priorities
            with torch.no_grad():
                # Calculate new errors for priority updates
                new_action_errors = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
                
                # For raise actions, include bet sizing errors
                if torch.any(raise_mask):
                    # Calculate normalized bet size errors for each sample
                    new_bet_errors = torch.zeros_like(new_action_errors)
                    
                    # Only add bet sizing errors for raise actions
                    raise_indices = torch.where(raise_mask)[0]
                    for i in raise_indices:
                        new_bet_errors[i] = F.smooth_l1_loss(
                            bet_size_predicts[i], bet_size_tensors[i], reduction='mean'
                        )
                    
                    # Combined error with smaller weight for bet sizing
                    combined_errors = new_action_errors + 0.5 * new_bet_errors
                else:
                    combined_errors = new_action_errors
                
                # [DEBUG 11] Check priority values
                if self.iteration % 100 == 0 and epoch == 0:
                    combined_errors_np = combined_errors.cpu().numpy()
                    max_priority = np.max(combined_errors_np) + 0.01
                    min_priority = np.min(combined_errors_np) + 0.01
                    mean_priority = np.mean(combined_errors_np) + 0.01
                    print(f"[DEBUG-PRIORITY] Priorities: min={min_priority:.2f}, max={max_priority:.2f}, mean={mean_priority:.2f}")
                
                # Update priorities (your existing code)
                combined_errors_np = combined_errors.cpu().numpy()
                for i, idx in enumerate(indices):
                    self.regret_memory.update_priority(idx, combined_errors_np[i] + 0.01)
            
            total_loss += combined_loss.item()
        
        # Return average loss
        return total_loss / epochs

    def train_strategy_net(self, batch_size=128, epochs=3):
        """
        Train the strategy network using collected samples.
        
        Args:
            batch_size: Size of training batches
            epochs: Number of training epochs per call
            
        Returns:
            Average training loss
        """
        if len(self.strategy_memory) < batch_size:
            return 0
        
        self.strategy_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch from memory
            batch = random.sample(self.strategy_memory, batch_size)
            states, strategies, bet_size_predicts, iterations = zip(*batch)
            
            # Convert to tensors
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_size_predicts)).unsqueeze(1).to(self.device)
            iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            
            # Weight samples by iteration (Linear CFR)
            weights = iteration_tensors / torch.sum(iteration_tensors)
            
            # Forward pass
            action_logits, bet_size_predicts = self.strategy_net(state_tensors)
            strategy_predicts = F.softmax(action_logits, dim=1)
            
            # Action type loss (weighted cross-entropy)
            # Add small epsilon to prevent log(0)
            action_loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(strategy_predicts + 1e-8), dim=1))
            
            # Bet size loss (only for states with raise actions)
            raise_mask = (strategy_tensors[:, 2] > 0)
            if raise_mask.sum() > 0:
                raise_indices = torch.nonzero(raise_mask).squeeze(1)
                raise_bet_predicts = bet_size_predicts[raise_indices]
                raise_bet_targets = bet_size_tensors[raise_indices]
                raise_weights = weights[raise_indices]
                
                # Use huber loss for bet sizing to be more robust to outliers
                bet_size_loss = F.smooth_l1_loss(raise_bet_predicts, raise_bet_targets, reduction='none')
                weighted_bet_size_loss = torch.sum(raise_weights * bet_size_loss.squeeze())
                
                # Combine losses with appropriate weighting
                combined_loss = action_loss + 0.5 * weighted_bet_size_loss
            else:
                combined_loss = action_loss
            
            # Backward pass and optimize
            self.strategy_optimizer.zero_grad()
            combined_loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=0.5)
            
            self.strategy_optimizer.step()
            
            total_loss += combined_loss.item()
        
        # Return average loss
        return total_loss / epochs

    def choose_action(self, state):
        """Choose an action for the given state during actual play."""
        legal_action_types = get_legal_action_types(state)
        
        if not legal_action_types:
            # Default to call if no legal actions (shouldn't happen)
            if pokers.ActionEnum.Call in state.legal_actions:
                return pokers.Action(pokers.ActionEnum.Call)
            elif pokers.ActionEnum.Check in state.legal_actions:
                return pokers.Action(pokers.ActionEnum.Check)
            else:
                return pokers.Action(pokers.ActionEnum.Fold)
            
        state_tensor = torch.FloatTensor(encode_state(state)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, bet_predicts = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            bet_multiplier = bet_predicts[0][0].item()
        
        # Filter to only legal actions
        legal_probs = np.array([probs[a] for a in legal_action_types])
        if np.sum(legal_probs) > 0:
            legal_probs = legal_probs / np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_action_types)) / len(legal_action_types)
        
        # Choose action based on probabilities
        action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
        action_type = legal_action_types[action_idx]
        
        # Use the predicted bet size for raise actions
        if action_type == 2:  # Raise
            return action_type_to_pokers_action(action_type, state, bet_multiplier)
        else:
            return action_type_to_pokers_action(action_type, state)

    def save_model(self, path_prefix):
        """Save the model to disk."""
        torch.save({
            'advantage_net': self.regret_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict()
        }, f"{path_prefix}_iteration_{self.iteration}.pt")
        
    def load_model(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.regret_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
