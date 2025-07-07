# train_sddpg.py
import rclpy
import threading
import time
import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from kineza_DroneLidarNavEnv_v2 import DroneLidarNavEnv  
from Kineza_SDDPG import SDDPG  # Fixed import name

def ros_spin_thread(node):
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            time.sleep(0.01)
    except Exception as e:
        print(f"Spin thread exception: {e}")

def evaluate_policy(env, agent, episodes=3):
    rewards = []
    for ep in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(obs, explore=False)
            print("Action:", action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    avg_reward = np.mean(rewards)
    print(f"Evaluation over {episodes} episodes: Avg Reward = {avg_reward:.2f}")
    return avg_reward

def compute_r_tag(prev_pos, curr_pos, goal_pos, d0=10.0):
    d_prev = np.linalg.norm(prev_pos - goal_pos)
    d_curr = np.linalg.norm(curr_pos - goal_pos)
    if d_curr > d0:
        return -30.0
    else:
        delta = d_prev - d_curr
        # Equation (11) from the paper
        return delta * np.exp(0.05 * delta)

def compute_r_obs(lidar_data, sigma=50.0, Dr=100.0):
    d = np.min(lidar_data)
    return sigma * (d / Dr - 1)

def main():
    rclpy.init()
    env = DroneLidarNavEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fixed state dimension - environment returns 18-dim observations
    agent = SDDPG(state_dim=18, action_dim=3, device=device)
    
    spin_thread = threading.Thread(target=ros_spin_thread, args=(env,), daemon=True)
    spin_thread.start()
    
    time.sleep(2.0)  # wait for data
    
    max_episodes = 1000
    max_steps = 600
    batch_size = 64
    update_every = 64
    save_path = "./sddpg_checkpoints"
    os.makedirs(save_path, exist_ok=True)
    writer = SummaryWriter(log_dir="sddpg_tensorboard")
    
    rewards_log = []
    step_count = 0
    for episode in range(max_episodes):
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.push((obs, action, reward, next_obs, float(done)))
            obs = next_obs
            episode_reward += reward
            step_count += 1
            if step_count % update_every == 0:
                agent.update(batch_size)
            if done:
                break
        
        rewards_log.append(episode_reward)
        writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Epsilon = {agent.epsilon:.3f}")
        
        # Evaluate policy every 20 episodes
        if (episode + 1) % 20 == 0:
            avg_eval = evaluate_policy(env, agent)
            writer.add_scalar('Eval/AvgReward', avg_eval, episode)
        
        # Save model every 50 episodes
        if (episode + 1) % 50 == 0:
            torch.save(agent.actor.state_dict(), os.path.join(save_path, f"actor_ep{episode+1}.pth"))
            torch.save(agent.critic.state_dict(), os.path.join(save_path, f"critic_ep{episode+1}.pth"))
            print(f"Checkpoint saved at episode {episode + 1}")
    
    torch.save(agent.actor.state_dict(), os.path.join(save_path, "actor_final.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(save_path, "critic_final.pth"))
    print("Training complete. Final model saved.")
    writer.close()
    env.close()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
