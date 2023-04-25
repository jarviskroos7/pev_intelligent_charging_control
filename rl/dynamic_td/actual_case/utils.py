import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '/Users/jarvis/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Spring 2023/CE290/pev_intelligent_charging_control/data')

def get_price(current_time, day=None, idx='12min'):

    if day:
        price = pd.read_csv(f'../../../data/price_day_{day}_idx.csv')['price'].values
    else:
        price = pd.read_csv(f'../../../data/price_day_idx_{idx}.csv')['price'].values
    
    try:
        return price[current_time]
    except Exception as e:
        print('incorrect time index:', e)
        return None

def roundSoc(x, prec=2, base=0.02):
  return round(base * round(float(x)/base), prec)