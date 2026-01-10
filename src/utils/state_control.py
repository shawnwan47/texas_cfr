import pokers

def get_legal_action_types(state):
    """Get the legal action types for the current state."""
    legal_action_types = []

    # Check each action type
    if pokers.ActionEnum.Fold in state.legal_actions:
        legal_action_types.append(0)

    if pokers.ActionEnum.Check in state.legal_actions or pokers.ActionEnum.Call in state.legal_actions:
        legal_action_types.append(1)

    if pokers.ActionEnum.Raise in state.legal_actions:
        legal_action_types.append(2)

    return legal_action_types

def action_type_to_pokers_action(action_type, state, bet_size_multiplier=None):
    """
    Convert action type and optional bet size to Pokers action.
    (Refined Raise Logic with Float Safeguard)
    """
    # Access VERBOSE, assuming it's set globally or accessible (e.g., self.verbose if it's an instance attr)
    # For this example, I'll assume VERBOSE is a global imported from your model.py or utils
    # If VERBOSE is an instance variable like self.verbose, use that instead.
    # from src.core.model import VERBOSE # Make sure this import is at the top of the file or VERBOSE is otherwise in scope

    try:
        if action_type == 0:  # Fold
            if pokers.ActionEnum.Fold in state.legal_actions:
                return pokers.Action(pokers.ActionEnum.Fold)
            # Fallback logic for Fold
            if pokers.ActionEnum.Check in state.legal_actions: return pokers.Action(pokers.ActionEnum.Check)
            if pokers.ActionEnum.Call in state.legal_actions: return pokers.Action(pokers.ActionEnum.Call)
            return pokers.Action(pokers.ActionEnum.Fold)  # Last resort

        elif action_type == 1:  # Check/Call
            if pokers.ActionEnum.Check in state.legal_actions:
                return pokers.Action(pokers.ActionEnum.Check)
            elif pokers.ActionEnum.Call in state.legal_actions:
                return pokers.Action(pokers.ActionEnum.Call)
            # Fallback logic for Check/Call
            if pokers.ActionEnum.Fold in state.legal_actions: return pokers.Action(pokers.ActionEnum.Fold)
            return pokers.Action(pokers.ActionEnum.Check)  # Last resort

        elif action_type == 2:  # Raise
            if pokers.ActionEnum.Raise not in state.legal_actions:
                # If Raise is not legal, fall back
                if pokers.ActionEnum.Call in state.legal_actions: return pokers.Action(pokers.ActionEnum.Call)
                if pokers.ActionEnum.Check in state.legal_actions: return pokers.Action(pokers.ActionEnum.Check)
                return pokers.Action(pokers.ActionEnum.Fold)

            player_state = state.players_state[state.current_player]
            current_bet = player_state.bet_chips  # What player already has in pot this round
            available_stake = player_state.stake  # Player's remaining chips

            call_amount = max(0.0, state.min_bet - current_bet)  # Additional chips needed to call

            min_raise_increment = 1.0
            if hasattr(state, 'bb') and state.bb is not None and float(state.bb) > 0:
                min_raise_increment = max(1.0, float(state.bb))
            elif state.min_bet > 0:  # If no BB, use min_bet if it implies a raise size
                # This part is a bit heuristic if BB is not well-defined.
                # The idea is that a raise should be somewhat meaningful.
                # If last bet was X, min_raise_increment is often X.
                # For simplicity, we'll stick to a small fixed minimum or BB.
                # A more robust way might involve looking at the previous raise amount.
                min_raise_increment = max(1.0, state.min_bet - current_bet if state.min_bet > current_bet else 1.0)

            # --- Initial Check: Can the player make ANY valid raise? ---
            # A valid raise means calling and then adding at least min_raise_increment.
            if available_stake < call_amount + min_raise_increment:
                if pokers.ActionEnum.Call in state.legal_actions:
                    return pokers.Action(pokers.ActionEnum.Call)
                else:
                    return pokers.Action(pokers.ActionEnum.Fold)
            # --- End Initial Check ---

            remaining_stake_after_call = available_stake - call_amount

            # Get target additional raise from network's bet_size_multiplier
            pot_size = max(1.0, state.pot)

            if bet_size_multiplier is None:
                bet_size_multiplier = 1.0  # Default if not provided
            else:
                bet_size_multiplier = float(bet_size_multiplier)
                # Optional: self.adjust_bet_size(state, bet_size_multiplier) if you use it

            bet_size_multiplier = max(0.1, min(3.0, bet_size_multiplier))
            network_desired_additional_raise = pot_size * bet_size_multiplier

            # Determine chosen_additional_amount based on network and game rules
            chosen_additional_amount = network_desired_additional_raise
            # Clip 1: Cannot raise more than all-in (remaining_stake_after_call)
            chosen_additional_amount = min(chosen_additional_amount, remaining_stake_after_call)
            # Clip 2: Must raise at least min_raise_increment
            chosen_additional_amount = max(chosen_additional_amount, min_raise_increment)

            # Safeguard: If due to clipping (e.g. min_raise_increment was > remaining_stake_after_call,
            # which shouldn't happen if initial check was correct), ensure it's not > remaining_stake_after_call.
            # This makes it an all-in if min_raise_increment forces it.
            if chosen_additional_amount > remaining_stake_after_call:
                chosen_additional_amount = remaining_stake_after_call

            # --- START: FLOAT SAFEGUARD ---
            total_chips_player_would_commit_this_turn = call_amount + chosen_additional_amount
            epsilon = 0.00001  # Tolerance for float comparisons

            if total_chips_player_would_commit_this_turn > available_stake + epsilon:
                chosen_additional_amount = available_stake - call_amount
                chosen_additional_amount = max(0.0, chosen_additional_amount)  # Ensure not negative
            # --- END: FLOAT SAFEGUARD ---

            # Ensure chosen_additional_amount is not negative after all adjustments
            chosen_additional_amount = max(0.0, chosen_additional_amount)

            return pokers.Action(pokers.ActionEnum.Raise, chosen_additional_amount)

        else:  # Should not be reached if action_type is 0, 1, or 2
            if pokers.ActionEnum.Call in state.legal_actions: return pokers.Action(pokers.ActionEnum.Call)
            if pokers.ActionEnum.Check in state.legal_actions: return pokers.Action(pokers.ActionEnum.Check)
            return pokers.Action(pokers.ActionEnum.Fold)

    except Exception as e:
        # Fall back to a safe action
        if hasattr(state, 'legal_actions'):
            if pokers.ActionEnum.Call in state.legal_actions: return pokers.Action(pokers.ActionEnum.Call)
            if pokers.ActionEnum.Check in state.legal_actions: return pokers.Action(pokers.ActionEnum.Check)
            if pokers.ActionEnum.Fold in state.legal_actions: return pokers.Action(pokers.ActionEnum.Fold)

    # Absolute last resort if state.legal_actions is not even available or empty
    return pokers.Action(pokers.ActionEnum.Fold)

def adjust_bet_size(state, base_multiplier):
    """
    Dynamically adjust bet size multiplier based on game state.

    Args:
        state: Current poker game state
        base_multiplier: Base bet size multiplier from the model

    Returns:
        Adjusted bet size multiplier
    """
    # Default adjustment factor
    adjustment = 1.0

    # Adjust based on game stage
    if int(state.stage) >= 2:  # Turn or River
        adjustment *= 1.2  # Increase bets in later streets

    # Adjust based on pot size relative to starting stack
    initial_stake = state.players_state[0].stake + state.players_state[0].bet_chips
    pot_ratio = state.pot / initial_stake
    if pot_ratio > 0.5:  # Large pot
        adjustment *= 1.1  # Bet bigger in large pots
    elif pot_ratio < 0.1:  # Small pot
        adjustment *= 0.9  # Bet smaller in small pots

    # Adjust based on position (more aggressive in late position)
    btn_distance = (state.current_player - state.button) % len(state.players_state)
    if btn_distance <= 1:  # Button or cutoff
        adjustment *= 1.15  # More aggressive in late position
    elif btn_distance >= 4:  # Early position
        adjustment *= 0.9  # Less aggressive in early position

    # Adjust for number of active players (larger with fewer players)
    active_players = sum(1 for p in state.players_state if p.active)
    if active_players <= 2:
        adjustment *= 1.2  # Larger bets heads-up
    elif active_players >= 5:
        adjustment *= 0.9  # Smaller bets multiway

    # Apply adjustment to base multiplier
    adjusted_multiplier = base_multiplier * adjustment

    # Ensure we stay within bounds
    return max(0.1, min(3.0, adjusted_multiplier))
