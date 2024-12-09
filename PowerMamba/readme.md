
The main script to be executed for running experiments is `run_longexp.py`. It coordinates the execution of various models and their hyperparameters across different datasets.

## How to use the scripts:

To run the experiments, you need to use the scripts located in the `scripts` folder. These scripts specify the parameters and configurations required for each model.

### Some Details of the Scripts:
- **Model Name** and **Dataset Directory**: These are the most critical elements inside each script. The features can be `M` which considers all the data and predicts for it; `s` will only consider the targeted data; and `Mm` will only predict for the last `c_out` column. If there is prediction data in your dataset, you should set `include_pred = 1`. The rest of the script contains hyperparameters.


### Running Scripts:

To run a script, use the following command in the current directory:

```bash
sh ./directory_to_scripts/scripts/script_name.sh
```

Replace `script_name.sh` with the name of the script you want to execute. For instance, to run the PowerMamba script, use:

```bash
sh ./scripts/PowerMamba.sh
```
