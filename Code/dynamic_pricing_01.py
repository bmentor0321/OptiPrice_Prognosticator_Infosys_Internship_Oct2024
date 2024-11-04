import pandas as pd
import os

from Utils.constants import RAW_DIR, file_name, RESULTS_DIR, output_file_name
from Utils.utils import get_mean

df = pd.read_csv(os.path.join(RAW_DIR, file_name))

hist_cost_ride_mean = get_mean(df, "Historical_Cost_of_Ride")

print(hist_cost_ride_mean)

tr_rmse = 0.8
val_rmse = 0.2
test_rmse = 0.4

df_metrics = pd.DataFrame(columns=['Model', "Model Description", "Train error", "Val error", "Test error"])

df_metrics.loc[len(df_metrics)] = ["Model 1", "All numerical features", tr_rmse, val_rmse, test_rmse]

if os.path.exists(os.path.join(RESULTS_DIR,output_file_name)):
    df_metrics_ = pd.read_csv(os.path.join(RESULTS_DIR,output_file_name))

    df_metrics_ = pd.concat([df_metrics_, df_metrics])

df_metrics_.to_csv(os.path.join(RESULTS_DIR, output_file_name), index=False)