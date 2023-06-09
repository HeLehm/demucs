import __main__

import time

from demucs import pretrained
import torch
from demucs.apply import apply_model
import argparse
import pandas as pd



def main(model_name, device='cpu', grad=False):
    # load model
    model = pretrained.get_model(model_name).to(device=device)
    # create input
    x = torch.randn(1, 2, 44100 * 30).to(device=device)  # 30 seconds of white noise for the test

    if grad:
        start = time.time()
        apply_model(model, x, device=device)
        end = time.time()
    else:
        start = time.time()
        with torch.no_grad():
            apply_model(model, x, device=device)
        end = time.time()

    total_time = end - start

    times = __main__.forward_times

    times = [times[i] - times[i - 1] for i in range(1, len(times))]
    names = __main__.forward_time_names

    data = {
        'model_name': model_name,
        'device': device,
        'grad': grad,
        'total_time': total_time,
        **dict(zip(names, times))
    }

    # turn times into pd dataframe
    times_df = pd.DataFrame(data, columns=['model_name', 'device', 'total_time', 'grad', *names], index=[0])
    return times_df




if __name__ == "__main__":
    models_to_test = ['mdx', 'htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'hdemucs_mmi', 'mdx_extra']

    dfs = []

    for model in models_to_test:
        for grad in [True, False]:
            for device in ["cpu", "mps"]:
                print(f"Testing {model} on {device} with grad {grad}")
                df = main(model, device=device, grad=grad)
                dfs.append(df)

    # save as csv
    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv("./demucs_performance.csv", index=False)


