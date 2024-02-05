import pandas as pd
import requests

def fetch_stock_data(file_path=None, api_url=None):
    if file_path:
        # when path is pricing data from CSV
        stock_data = pd.read_csv(file_path)
    elif api_url:
        #when path is pricing data from API
        response = requests.get(api_url)
        stock_data = pd.DataFrame(response.json())
    else:
        raise ValueError("Either file_path or api_url must be provided")
    return stock_data

#the actual black sholes in ppython
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return C

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    P = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return P

def price_options(stock_data, K, T, r, sigma, option_type='call'):
    S = stock_data['Close'].iloc[-1]
    if option_type == 'call':
        C = black_scholes_call(S, K, T, r, sigma)
    elif option_type == 'put':
        P = black_scholes_put(S, K, T, r, sigma)
    else:
        raise ValueError("Invalid option_type")
    return C, P

# alternative code to fetch data from your CSV
stock_data = fetch_stock_data(file_path='stock_data.csv')

# to return a call or put value from black scholes
K =