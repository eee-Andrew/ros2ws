import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from PPO_DroneLidarNavEnv_v4 import DroneLidarNavEnv
import threading
import time

def ros_spin_thread(node):
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            time.sleep(0.05)
    except Exception as e:
        print(f"Spin thread exception: {e}")


def main():
    # Initialize ROS
    try:
        # Create environment
        print("Creating environment...")
        env = DroneLidarNavEnv()
        
        # Debug: List available topics
        print("Available topics:")
        topic_list = env.get_topic_names_and_types()
        drone_topics = [t for t in topic_list if ('drone' in t[0].lower() or 'odom' in t[0].lower())]
        for topic, msg_type in drone_topics:
            print(f"  {topic}: {msg_type[0]}")
        
        if not drone_topics:
            print("  No drone or odom topics found!")
            all_topics = [topic for topic, _ in topic_list]
            print(f"  All available topics: {all_topics[:10]}...")  # Show first 10
        
        # Start ROS spinning in separate thread
        spin_thread = threading.Thread(target=ros_spin_thread, args=(env,), daemon=True)
        spin_thread.start()
        
        # Check environment (optional but recommended)
        print("Checking environment...")
        check_env(env)
        
        # Wait for initial data
        print("Waiting for initial sensor data...")
        time.sleep(3.0)
        
        # Create PPO model
        print("Creating PPO model...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./ppo_drone_tensorboard/"
        )
        
        # Create checkpoint callback to save model periodically
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./ppo_drone_checkpoints/",
            name_prefix="ppo_drone"
        )
        
        # Train the model
        print("Starting training...")
        model.learn(
            total_timesteps=100000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        model.save("ppo_drone_final")
        print("Training completed! Model saved as 'ppo_drone_final'")
        
        # Test the trained model
        print("Testing trained model...")
        env = DroneLidarNavEnv()

        # Wait for initial sensor data to stabilize
        print("Waiting for initial sensor data...")
        time.sleep(3.0)

        # Reset environment first (reset position)
        obs, info = env.reset()

        # Then takeoff command
        print("Taking off...")
        env.takeoff()

        # Give the drone some time to take off before training starts
        time.sleep(3.0)
        episode_count = 0
        for i in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_time = info.get('episode_time', 0)
            distance_to_goal = info.get('distance_to_goal', 0)
            timeout = info.get('timeout', False)
            
            print(f"Step {i}: Time: {episode_time:.1f}s, Distance: {distance_to_goal:.2f}m, Reward: {reward:.2f}")
            
            if done:
                episode_count += 1
                reason = "TIMEOUT" if timeout else ("SUCCESS" if reward > 50 else "COLLISION/OOB")
                print(f"Episode {episode_count} finished after {i+1} steps ({episode_time:.1f}s) - {reason}")
                if episode_count >= 3:  # Test 3 episodes
                    break
                obs, info = env.reset()
                i = 0
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Clean shutdown
        print("Shutting down...")
        if 'env' in locals():
            env.close()
        rclpy.shutdown()

if __name__ == "__main__":
    main()