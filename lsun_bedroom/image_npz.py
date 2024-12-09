import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

df  = pd.read_csv("/data2/ctm_model/LSUN_Bedroom_Output/ctm_bs_512/progress.csv")

lst_value1 = [0.334084216, 0.050933401,	0.948485108, 0.809671388, 59.84496723, 9.2495, 936.1426529, 256512, 500]
lst_value2 = [0.225401916, 0.04920471, 0.998657867, 0.836021385, 40.87773986, 9.7495, 936.1586388,	512512,	1000]
lst_keys = ["consistency_loss mean", "consistency_loss std", "denoising_loss mean",	"denoising_loss std", "grad_norm", \
"lg_loss_scale", "param_norm", "samples", "step"]

# new_row = {}
# new_row2 = {}

# for i in range(len(lst_keys)):
#     new_row[lst_keys[i]] = lst_value1[i]
#     new_row2[lst_keys[i]] = lst_value2[i]

# df = df.append(new_row, ignore_index=True)
# df = df.append(new_row2, ignore_index=True)
# df = df.sort_values(by='step')
print(df.head())

# Plot dots and connect them with a line
plt.plot(df["step"], df["consistency_loss mean"], 'bo-') 
plt.xlabel('Iterations') # Labels the x-axis
plt.ylabel('Consistency loss') # Labels the y-axis
plt.savefig("ctm_loss_graph.png")
plt.close()

# Plot dots and connect them with a line
plt.plot(df["step"], df["denoising_loss mean"], 'go-') 
plt.xlabel('Iterations') # Labels the x-axis
plt.ylabel('DSM loss') # Labels the y-axis
plt.savefig("dsm_loss_graph.png")
plt.close()