import pandas as pd
from stable_baselines3 import PPO
from trading_env import StockTradingEnv

# Load dữ liệu test
df_test = pd.read_csv('data/data_test.csv')

# Load mô hình đã huấn luyện
model = PPO.load("ppo_stock_trading_optimized")

# Tạo môi trường test
env = StockTradingEnv(df_test)
obs, _ = env.reset()

done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()

env.close()
print("✅ Kiểm tra hoàn tất!")
