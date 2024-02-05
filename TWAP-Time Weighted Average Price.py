import numpy as np
import pandas as pd
from scipy.stats import norm

class TWAPAlgo:
    def __init__(self, start_time, end_time, initial_inventory, initial_price, gbm_data, garch_model, trade_threshold):
        self.start_time = start_time
        self.end_time = end_time
        self.initial_inventory = initial_inventory
        self.initial_price = initial_price
        self.gbm_data = gbm_data
        self.garch_model = garch_model
        self.trade_threshold = trade_threshold
        self.current_time = start_time
        self.current_inventory = initial_inventory
        self.current_price = initial_price
        self.trade_times = []

    def calculate_price(self):
        """
        Calculate the current price based on the geometric Brownian motion model.
        """
        drift = self.gbm_data['drift'].iloc[self.current_time]
        sigma = self.gbm_data['sigma'].iloc[self.current_time]
        z = np.random.normal()
        price = self.current_price * np.exp((drift + 0.5 * sigma ** 2) * (1 / 252) + sigma * np.sqrt(1 / 252) * z)
        return price

    def calculate_volatility(self):
        """
        Calculate the current volatility based on the GARCH model.
        """
        volatility = self.garch_model.forecast(self.current_time)['sigma2'].values[0]
        return volatility

    def check_trade(self):
        """
        Check if a trade should be placed based on the current price and volatility.
        """
        if np.abs(self.calculate_price() - self.current_price) > self.trade_threshold * self.calculate_volatility():
            self.trade_times.append(self.current_time)
            self.current_price = self.calculate_price()
            self.current_inventory = self.initial_inventory
            self.current_time += 1

    def simulate(self):
        """
        Simulate the TWAP algorithm.
        """
        while self.current_time < self.end_time:
            self.check_trade()

        time_index = pd.date_range(start=self.start_time, end=self.end_time, freq='B')
        self.trade_times = [time_index[t] for t in self.trade_times]
        trades = pd.DataFrame({'trade_time': self.trade_times,
                               'inventory': self.initial_inventory,
                               'price': self.current_price})
        twap = trades.set_index('trade_time').resample('D').mean()['price']
        return twap

#How to use with multivariate GARCH
# link data path from your GARCH model and/or geometric Brownian motion model. 
gbm_data = pd.read_csv('gbm_data.csv')
garch_model = pd.read_pickle('garch_model.pkl')

# Set the TWAP algos initial parameters
start_time = 0
end_time = 252
initial_inventory = 100
initial_price = 100
trade_threshold = 0.01

algo = TWAPAlgo(start_time, end_time, initial_inventory, initial_price, gbm_data, garch_model, trade_threshold)

# Run that shit !
twap = algo.simulate()
print(twap)
