import colorama
import pandas as pd
import numpy as np
import sys
from colorama import Fore, Back, Style

def main():
    colorama.init()
    max_length = [len(name) for name in sys.argv[1:]]
    for file in sys.argv[1:]:
        df = pd.read_csv(file, header=None)
        time = (df[1] - df[0]) / df[2]

        file_name = (file + " ").ljust(max(max_length) + 2, '.')

        print(f"{Fore.LIGHTBLACK_EX}{file_name} {Style.RESET_ALL}", end="")
        q1 = np.percentile(time, 25)
        q3 = np.percentile(time, 75)
        print(f"{np.percentile(time, 5):>10.1f}  {time.mean():>10.1f}  {np.percentile(time, 95):>10.1f}  [IQR:{q3 - q1:>10.1f}]")

if __name__ == "__main__":
    main();
