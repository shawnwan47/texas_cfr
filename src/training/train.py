# src/training/train.py
import time
import os
import random
import glob
import torch
import numpy as np
import pokers
from torch.utils.tensorboard import SummaryWriter
from src.core.deep_cfr import DeepCFRAgent
from src.core.model import set_verbose, encode_state
from src.utils.logging import log_game_error
from src.utils.settings import STRICT_CHECKING, set_strict_checking
from src.utils.state_control import get_legal_action_types
from src.agents.random_agent import RandomAgent


def load_random_opponents(num_players):
    return [RandomAgent(i) for i in range(num_players)]

def load_existing_opponents(num_players, checkpoint):
    # Create the full opponent list
    opponent_list = []  # Initialize with None
    # Fill any remaining positions with random agents
    for i in range(num_players):
        agent_tmp = DeepCFRAgent(player_id=i, num_players=num_players, device=device)
        agent_tmp.load_model(checkpoint)
        opponent_list.append(agent_tmp)
    return opponent_list

def load_mixed_agents(checkpoint_dir, model_prefix, num_players):
    # Get all checkpoint files with the specified prefix
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, f"{model_prefix}*.pt"))

    if not checkpoint_files:
        print(f"WARNING: No checkpoint files found with prefix '{model_prefix}' in {checkpoint_dir}")
        # Create random agents as fallback
        return [RandomAgent(i) for i in range(num_players)]

    # Select random checkpoints
    selected_files = random.sample(checkpoint_files, min(num_players, len(checkpoint_files)))
    print(f"Selected checkpoints: {[os.path.basename(f) for f in selected_files]}")

    # Create agents from the selected checkpoints
    selected_agents = []

    # Always keep a random agent at position 1 (optional, can be modified)
    random_agent = RandomAgent(1)

    # Load checkpoint agents for other positions
    current_pos = 1
    for checkpoint_file in selected_files:
        # Skip position 0 as it's reserved for our learning agent
        if current_pos == 0:
            current_pos += 1

        # Create and load agent
        checkpoint_agent = DeepCFRAgent(player_id=current_pos, num_players=6, device=device)
        checkpoint_agent.load_model(checkpoint_file)

        # Add to list
        selected_agents.append((current_pos, checkpoint_agent))
        current_pos += 1
        if current_pos >= 6:
            current_pos = 1  # Wrap around, skipping position 0

    # Create the full opponent list
    opponents = [None] * 6  # Initialize with None

    # Set position 0 to None (will be our learning agent)
    opponents[0] = None

    # Set the random agent at position 1
    opponents[1] = random_agent

    # Fill in the checkpoint agents
    for pos, agent in selected_agents:
        if pos != 1:  # Skip position 1 as it's already the random agent
            opponents[pos] = agent

    # Fill any remaining positions with random agents
    for i in range(6):
        if opponents[i] is None and i != 0:
            opponents[i] = RandomAgent(i)

    return opponents

def evaluate(agent_eval, opponents_eval, num_games=500):
    """
    Evaluate the trained agent against opponent agents.
    Each agent will receive and process observations from its own perspective.
    """
    player_id = agent_eval.player_id
    num_players = len(opponents_eval)
    total_profit = 0
    completed_games = 0

    # Wrap checkpoint agents
    for game in range(num_games):
        try:
            # Create a new poker game with rotating button
            state = pokers.State.from_seed(
                n_players=num_players,
                button=game % num_players,  # Rotate button for fairness
                sb=1,
                bb=2,
                stake=200.0,
                seed=game + 42195 # Using different seeds than training
            )

            # Play until the game is over
            while not state.final_state:
                if player_id == state.current_player:
                    action = agent_eval.choose_action(state)
                else:
                    action = opponents[state.current_player].choose_action(state)
                # Apply the action with conditional status check
                new_state = state.apply_action(action)
                if new_state.status != pokers.StateStatus.Ok:
                    log_file = log_game_error(state, action, f"State status not OK ({new_state.status})")
                    if STRICT_CHECKING:
                        raise ValueError(f"State status not OK ({new_state.status}). Details logged to {log_file}")
                    else:
                        print(f"WARNING: State status not OK ({new_state.status}) in game {game}. Details logged to {log_file}")
                        break  # Skip this game in non-strict mode
                state = new_state

            # Only count completed games
            if state.final_state:
                # Add the profit for this game
                profit = state.players_state[player_id].reward
                total_profit += profit
                completed_games += 1

        except Exception as e:
            if STRICT_CHECKING:
                raise  # Re-raise the exception in strict mode
            else:
                print(f"Error in game {game}: {e}")
                # Continue with next game in non-strict mode

    # Return average profit only for completed games
    if completed_games == 0:
        print("WARNING: No games completed during evaluation!")
        return 0

    return total_profit / completed_games

def train_cfr_oppo(agent_train, opponents_choose, iterations, traversals, save_dir, log_dir):
    """
    Train a Deep CFR agent in a 6-player No-Limit Texas Hold'em game
    against 5 random opponents.
    """
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)

    num_players = len(opponents_choose)

    # For tracking learning progress
    losses = []
    profits = []

    # ADDED: Initial evaluation before training begins
    print("Initial evaluation...")
    initial_profit = evaluate(agent_train, opponents_choose, num_games=500)
    profits.append(initial_profit)
    print(f"Initial average profit per game: {initial_profit:.2f}")
    writer.add_scalar('Performance/Profit', initial_profit, 0)

    # Checkpoint frequency
    checkpoint_frequency = 100  # Save a checkpoint every 100 iterations

    # Training loop
    for iteration in range(1, iterations + 1):
        agent_train.iteration = iteration
        start_time = time.time()

        print(f"Iteration {iteration}/{iterations}")

        # Run traversals to collect data
        print("  Collecting data...")
        for traversal in range(traversals):
            # Create a new poker game
            state = pokers.State.from_seed(
                n_players=num_players,
                button=traversal % num_players,
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, traversal + 42195)
            )

            # Perform CFR traversal
            agent_train.cfr_traverse(state, iteration, opponents_choose)

        # Track traversal time
        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)

        # Train advantage network
        print("Training cfr network...")
        adv_loss = agent_train.train_regret_net()
        losses.append(adv_loss)
        print(f"Regret network loss: {adv_loss:.6f}")

        # Log the loss to tensorboard
        writer.add_scalar('Loss/Regret', adv_loss, iteration)
        writer.add_scalar('Memory/Regret', len(agent_train.regret_memory), iteration)

        # Every few iterations, train the strategy network and evaluate
        if iteration % 10 == 0 or iteration == iterations:
            print("Training strategy network...")
            strat_loss = agent_train.train_strategy_net()
            print(f"Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar('Loss/Strategy', strat_loss, iteration)

            # Evaluate the agent
            print("Evaluating agent...")
            avg_profit = evaluate(agent_train, opponents_choose, num_games=500)
            profits.append(avg_profit)
            print(f"Average profit per game: {avg_profit:.2f}")
            writer.add_scalar('Performance/Profit', avg_profit, iteration)

        # Save checkpoint periodically
        if iteration % checkpoint_frequency == 0:
            checkpoint_path = f"{save_dir}/checkpoint_iter_{iteration}.pt"
            torch.save({
                'iteration': iteration,
                'regret_net': agent_train.regret_net.state_dict(),
                'strategy_net': agent_train.strategy_net.state_dict(),
                'losses': losses,
                'profits': profits
            }, checkpoint_path)

        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print(f"  Regret memory size: {len(agent_train.regret_memory)}")
        print(f"  Strategy memory size: {len(agent_train.strategy_memory)}")
        writer.add_scalar('Memory/Strategy', len(agent_train.strategy_memory), iteration)

        # Commit the tensorboard logs
        writer.flush()
        print()

    # Final evaluation
    print("Final evaluation...")
    avg_profit = evaluate(agent_train, opponents_choose, num_games=1000)
    print(f"Final performance: Average profit per game: {avg_profit:.2f}")
    writer.add_scalar('Performance/FinalProfit', avg_profit, 0)

    # Close the tensorboard writer
    writer.close()

    return agent_train, losses, profits

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent for poker')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--num-players', type=int, default=2, help='Number of players')
    parser.add_argument('--self-play', action='store_true', help='Train against checkpoint instead of random agents')
    parser.add_argument('--mixed', action='store_true', help='Train against mixed checkpoints')
    parser.add_argument('--refresh-interval', type=int, default=1000, help='Interval to refresh opponent pool')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to continue training from')
    parser.add_argument('--checkpoint-dir', type=str, default='models', help='Directory containing checkpoint models')
    parser.add_argument('--model-prefix', type=str, default='t_', help='Prefix for models to include in selection pool')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/deepcfr', help='Directory for tensorboard logs')
    parser.add_argument('--strict', action='store_true', help='Enable strict error checking that raises exceptions for invalid game states')
    args = parser.parse_args()

    # Strict training for debug
    set_strict_checking(args.strict)
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set verbosity
    set_verbose(args.verbose)

    # Create the directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(args.log_dir)
    # Initialize the learning agent
    agent = DeepCFRAgent(player_id=0, num_players=args.num_players, device=device)
    suffix = ''
    if args.checkpoint:
        suffix += '_finetune'
        agent.load_model(args.checkpoint)
    else:
        suffix += '_base'
    if args.self_play and args.checkpoint:
        suffix += '_selfplay'
        opponents = load_existing_opponents(num_players=args.num_players, checkpoint=args.checkpoint)
    else:
        suffix += '_random'
        opponents = load_random_opponents(num_players=args.num_players)

    print(f"Starting Deep CFR training for {args.iterations} iterations")
    print(f"Using {args.traversals} traversals per iteration")
    print(f"Logs will be saved to: {args.log_dir}")
    print(f"Models will be saved to: {args.save_dir}")

    # Train the Deep CFR agent
    agent, losses, profits = train_cfr_oppo(
        agent_train=agent,
        opponents_choose=opponents,
        iterations=args.iterations,
        traversals=args.traversals,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )

    print(f"Checkpoint saved to {args.save_dir}")
    model_path = f"{args.save_dir}/cfr{suffix}.pt"
    agent.save_model(model_path)
    print(f"Model saved to {model_path}")
    print("\nTraining Summary:")
    print(f"Final loss: {losses[-1]:.6f}")
    if profits:
        print(f"Final average profit: {profits[-1]:.2f}")
    print("\nTo view training progress:")
    print(f"Run: tensorboard --logdir={args.log_dir}")
    print("Then open http://localhost:6006 in your browser")