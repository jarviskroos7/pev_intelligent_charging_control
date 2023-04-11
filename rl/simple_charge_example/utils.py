import numpy as np
import pandas as pd

def get_price(current_time, day=None):

    if day:
        price = pd.read_csv(f'../data/price_day_{day}_idx.csv')['price'].values
    else:
        price = pd.read_csv('../data/price_day_idx.csv')['price'].values
    
    try:
        return price[current_time]
    except Exception as e:
        print('incorrect time index:', e)
        return None

