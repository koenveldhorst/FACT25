import argparse
import pandas as pd
import re
import numpy as np
import os

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--logs_folder", type = str, default = "emissions", help="result") #celeba, waterbird
    parser.add_argument("--logs_file", type = str, default = "waterbirds", help="result") #celeba, waterbird
    parser.add_argument("--save_result", default = False)
    args = parser.parse_args()
    return args


def main(args):
    # logs_dir = args.logs_prefix + ".csv"
    logs_dir = args.logs_folder + "/" + args.logs_file

    if not os.path.exists(logs_dir):
        print("Error: File not found")
        return

    df = pd.read_csv(logs_dir, sep=';')
    keys = df.keys()
    pattern = "POWER_W"
    power_usage_keys = [s for s in keys if pattern in s]

    power_usage = np.array(df[power_usage_keys])

    total_power_usage_per_second_per_component = df[power_usage_keys].mean(axis=0)
    print("Power usage (per second, per component in W):")
    print(total_power_usage_per_second_per_component)

    total_power_usage_per_second = power_usage.mean(axis=0).sum()
    print("Power usage:", total_power_usage_per_second, "W")

    time = np.array(df['ELAPSED'])[-1]
    total_power_usage = total_power_usage_per_second * time
    print("Total power usage:", total_power_usage, "J")

    print("Total power usage:", total_power_usage_per_second * (time / 3600) / 1000, "kWh")


if __name__ == "__main__":
    args = parse_args()
    main(args)
