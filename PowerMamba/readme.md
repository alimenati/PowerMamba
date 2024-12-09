# Running the Main Script: `run_longexp.py`

The main script to be executed for running experiments is `run_longexp.py`. It coordinates the execution of various models and their hyperparameters across different datasets.

## Using Scripts in the `scripts` Directory

To run the experiments, you need to use the scripts located in the `scripts` folder. These scripts specify the parameters and configurations required for each model.

### Running Scripts

To run a script, use the following command in the current directory:

```bash
sh ./directory_to_scripts/scripts/script_name.sh
```

Replace `script_name.sh` with the name of the script you want to execute. For instance, to run the PowerMamba script, use:

```bash
sh ./scripts/PowerMamba.sh
```
