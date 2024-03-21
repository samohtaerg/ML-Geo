# --- IMPORTS ---
import autograd.numpy as np  # Thinly-wrapped version of Numpy
from joblib import Parallel, delayed
import csv

# call functions
from No_outlier_task import our_task
from Outlier_task import our_task_outlier

# --- SEEDING ---
np.random.seed(0)

# --- PARAMETERS ---
h_low = 0
h_high = 0.2
h_step = 0.1
parallel_num = 4

# --- 1.1 MTL_OURS ALGORITHM ---
# --- 1.2 MTL ALGORITHM CLASSIC (SHARED REPRESENTATION) ---
# --- 1.3 DISTANCE FUNCTIONS ---
# --- 2.1 DATA GENERATION AND EVAL (NO OUTLIER) ---
# --- 2.2 DATA GENERATION AND EVAL (OUTLIER) ---

# --- MAIN EXECUTION ---
h_list = np.arange(h_low, h_high, h_step)

mse_no_outlier = np.zeros((h_list.size, 4))
mse_no_outlier = np.array(Parallel(n_jobs=parallel_num)(delayed(our_task)(h) for h in h_list))

mse_outlier = np.zeros((h_list.size, 4))
mse_outlier = np.array(Parallel(n_jobs=parallel_num)(delayed(our_task_outlier)(h) for h in h_list))

mse_no_outlier = mse_no_outlier.reshape((1, h_list.size * 4))
mse_outlier = mse_outlier.reshape((1, h_list.size * 4))

with open("C:/Users/samoh/PycharmProjects/MTLa/MTL_Geo_Output/demo.py" + str(0) + "_no_outlier_test_01.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(mse_no_outlier)

with open("C:/Users/samoh/PycharmProjects/MTLa/MTL_Geo_Output/demo.py" + str(0) + "_outlier_test_01.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(mse_outlier)
