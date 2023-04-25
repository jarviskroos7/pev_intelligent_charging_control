# a = [1,2,3,4,5]
# with open('output.txt', 'a') as f:
#     f.write("%s\n "%a)
# import pandas as pd
# winter_emission = pd.read_csv("pred_may.csv")
# print(len(winter_emission["pred"].to_list()))

import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0.01,1,20)
y = 193*np.log(x) + 14587
plt.plot(x,y)
plt.show()