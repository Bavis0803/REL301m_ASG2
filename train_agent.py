import pandas as pd
from stable_baselines3 import PPO
from trading_env import StockTradingEnv

# Load dữ liệu lịch sử giá của AVGO
df = pd.read_csv('data/data_train.csv')

# Tạo môi trường RL
env = StockTradingEnv(df)

# Khởi tạo mô hình PPO
model = PPO(
    "MlpPolicy", env,
    learning_rate=1e-4,  # Tăng learning rate để cập nhật nhanh hơn
    n_steps=4096,  # Học xu hướng dài hơn
    batch_size=128,  # Batch lớn hơn để giảm overfitting
    gamma=0.99,  # Quan tâm đến phần thưởng dài hạn
    clip_range=0.3,  # Tăng để PPO cập nhật policy mạnh hơn
    ent_coef=0.02,  # Tăng exploration
    vf_coef=0.4,  # Giảm phụ thuộc vào mạng giá trị
    verbose=1
)

# Huấn luyện mô hình
model.learn(total_timesteps=500000)

# Lưu mô hình đã train
model.save("ppo_stock_trading_optimized")
