
The main script to be executed for running experiments is `run_longexp.py`. It coordinates the execution of various models and their hyperparameters across different datasets.

## How to use the scripts:

To run the experiments, you need to use the scripts located in the `scripts` folder. These scripts specify the parameters and configurations required for each model.

### Some Details of the Scripts:
- **Model Name** and **Dataset Directory**: These are the most critical elements inside each script. The features can be `M` which considers all the data and predicts for it; `s` will only consider the targeted data; and `Mm` will only predict for the last `c_out` column. If there is prediction data in your dataset, you should set `include_pred = 1`. The rest of the script contains hyperparameters.
#### `dictionaries` Directory Inside the `scripts` Directory:
For each dataset, you can define a dictionary which contains the following information:
- **`project_dict`**: If you have predictions for some of your columns, indicate it here. The key in the dictionary is the column name, and the values represent:
  - The historical data for that column.
  - The first column number where predictions start for that specific column. You can leave this as `{}` if you are not incorporating external predictions.
  - For example, in the following image, the first number in the dictionary should be the column number for the wind column, and the second one is the column number for the first prediction column, which corresponds to '1h pred'.

<div style="text-align: center; margin-top: 20px;">
    <div style="display: inline-block; text-align: center;">
        <img src="/main/pics/time_series.png" alt="Performance Results" style="width:400px; height:400px;">
        <p>An Example of how to incorporate the external prediction. For more information, please see the paper.</p>
    </div>
</div>


- **`Col_info_dict`**: Useful for reporting partial MSE. If you have columns that can be categorized (e.g., they are all prices for different regions or loads for different regions), specify them in this dictionary. The format is:
  - The first value is the index where that group of columns starts.
  - The second value is the number of columns in that group.
  - Ensure that columns belonging to the same group are adjacent to each other. If you don’t want partial MSE, leave it as `{}`.

### Running Scripts:

To run a script, use the following command in the current directory:

```bash
sh ./directory_to_scripts/scripts/script_name.sh
```

Replace `script_name.sh` with the name of the script you want to execute. For instance, to run the PowerMamba script, use:

```bash
sh ./scripts/PowerMamba.sh
```
