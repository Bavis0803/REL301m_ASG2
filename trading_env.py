import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000, total_assets_threshold=1000, no_trade_penalty=3072, transaction_cost=0.0025):
        super(StockTradingEnv, self).__init__()

        # Dữ liệu giá lịch sử (OHLC)
        self.df = df
        self.current_step = 0

        # Các biến trạng thái
        self.initial_balance = initial_balance
        self.balance = initial_balance  # Tiền mặt
        self.shares_held = 0  # Số lượng cổ phiếu đang nắm giữ
        self.total_assets_threshold = total_assets_threshold
        self.no_trade_penalty = no_trade_penalty
        self.last_trade_step = 0  # Đếm số phiên không giao dịch
        self.trades = []  # Lưu lại lịch sử giao dịch
        self.transaction_cost = transaction_cost  # Phí giao dịch (0.25%)

        # Không gian hành động: [0]: Giữ, [1]: Mua, [2]: Bán
        self.action_space = spaces.Discrete(3)

        # Không gian trạng thái (OHLC, Balance, Shares Held)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(6,), dtype=np.float32
        )

    def _calculate_buy_hold_return(self):
        """Tính lợi nhuận của chiến lược Buy & Hold (mua ngày đầu, bán ngày cuối)"""
        if len(self.df) < 2:
            return 0  # Tránh lỗi nếu dữ liệu quá ít

        initial_price = self.df.iloc[0]['Close']
        final_price = self.df.iloc[-1]['Close']
        return np.log(final_price) - np.log(initial_price)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.df.empty:
            raise ValueError("❌ Lỗi: DataFrame đầu vào bị rỗng!")

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.last_trade_step = 0
        self.trades = []
        self.history_log = []
        self.rbh = self._calculate_buy_hold_return()

        return self._next_observation(), {}

    def _next_observation(self):
        """Trả về trạng thái gồm giá OHLC hiện tại, tiền mặt, và số cổ phiếu đang nắm giữ"""
        obs = np.array([
            self.df.iloc[self.current_step]['Open'],
            self.df.iloc[self.current_step]['High'],
            self.df.iloc[self.current_step]['Low'],
            self.df.iloc[self.current_step]['Close'],
            self.balance,
            self.shares_held
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            done = True
            return self._next_observation(), 0, done, False, {}

        self.current_step += 1
        current_price = self.df.iloc[self.current_step]['Close']

        previous_assets = self.balance + \
            (self.shares_held * self.df.iloc[self.current_step - 1]['Close'])

        # Thực hiện hành động
        if action == 1:  # Mua cổ phiếu
            buy_amount = np.random.uniform(0.39, 0.75) * self.balance
            num_shares = buy_amount // current_price
            cost = num_shares * current_price * \
                (1 + self.transaction_cost)  # Tính phí mua

            if cost > 0 and self.balance >= cost:
                self.balance -= cost
                self.shares_held += num_shares
                self.trades.append(
                    {'step': self.current_step, 'type': 'buy', 'price': current_price, 'shares': num_shares})

        elif action == 2:  # Bán cổ phiếu
            sell_amount = np.random.uniform(0.23, 0.75) * self.shares_held
            revenue = sell_amount * current_price * \
                (1 - self.transaction_cost)  # Tính phí bán

            if sell_amount > 0 and self.shares_held >= sell_amount:
                self.balance += revenue
                self.shares_held -= sell_amount
                self.trades.append({'step': self.current_step, 'type': 'sell',
                                   'price': current_price, 'shares': sell_amount})

        # Tính tổng tài sản
        current_assets = self.balance + (self.shares_held * current_price)

        # Tính reward dựa trên lợi nhuận cộng dồn
        reward = self._calculate_excess_return(previous_assets, current_assets)

        # Kiểm tra điều kiện thắng/thua
        done = current_assets >= 1000000 or current_assets < 1000 or self.balance <= -5000

        return self._next_observation(), reward, done, False, {}

    def _calculate_excess_return(self, previous_assets, current_assets):
        """Tính phần thưởng dựa trên lợi nhuận cộng dồn so với Buy & Hold"""
        # Tránh lỗi khi tài sản về 0
        previous_assets = max(previous_assets, 1)
        current_assets = max(current_assets, 1)

        # Tính lợi nhuận giao dịch của agent
        r = np.log(current_assets) - np.log(previous_assets)

        # Tính lợi nhuận Buy & Hold
        excess_return = r - self.rbh

        return excess_return * 100  # Nhân 100 để tăng giá trị reward

    def render(self):
        """Hiển thị trạng thái hiện tại và ghi log"""
        total_assets = self.balance + \
            (self.shares_held * self.df.iloc[self.current_step]['Close'])

        log_message = (
            f"[Step {self.current_step}] 💰 Tiền mặt: {self.balance:.2f} USD | "
            f"📈 Cổ phiếu: {self.shares_held} | 🏦 Tổng tài sản: {total_assets:.2f} USD"
        )

        print(log_message)
        self.history_log.append(log_message)

    def close(self):
        """Đóng môi trường và hiển thị tổng kết giao dịch"""
        final_price = self.df.iloc[self.current_step]['Close']
        total_assets = self.balance + (self.shares_held * final_price)

        summary_message = (
            "\n===== 📊 TỔNG KẾT GIAO DỊCH =====\n"
            f"🔹 Số bước giao dịch: {self.current_step}\n"
            f"🔹 Số cổ phiếu còn lại: {self.shares_held}\n"
            f"🔹 Tiền mặt còn lại: {self.balance:.2f} USD\n"
            f"🔹 Giá trị tổng tài sản (quy đổi cổ phiếu): {total_assets:.2f} USD\n"
            "==================================\n"
        )

        print(summary_message)
        self.history_log.append(summary_message)

        with open("history_log.txt", "w", encoding="utf-8") as log_file:
            for line in self.history_log:
                log_file.write(line + "\n")

        if self.trades:
            trade_log = pd.DataFrame(self.trades)
            trade_log.to_csv("trade_log.csv", index=False)
            print("📁 Log giao dịch đã được lưu tại trade_log.csv")
