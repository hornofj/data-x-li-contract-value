# How to run this


1. Clone this repo
2. Make sure you have the required libraries installed (preferably with pip): `pandas`, `numpy`, `lightgbm`, `sklearn`
3. Create empty directory `data-x-li-data` at the same level as the directory where you cloned this repo. It will be used for source CSVs and generated outputs so they won't be committed to this repo.
4. Run `ir_nss.py`. This will generate the file `nss.csv` in the `data-x-li-data` directory. Be patient, it takes about two minutes.
5. Put the following files (you can find them zipped in MS Teams) in the `data-x-li-data` directory (keep the filenames lowercase to avoid errors on some systems):
   - `scen_0001-0200.csv`
   - `scen_0201-0500.csv`
   - `scen_0501-0700.csv`
   - `scen_0701-1000.csv`
   - `scen_1001-1300.csv`
   - `scen_1301-1600.csv`
   - `scen_1601-1900.csv`
   - `scen_1901-2236.csv`
6. Run `data_load.py`. This will read the source CSVs and `nss.csv` file and generate `df_merged_train_test.pickle`, again in the `data-x-li-data` directory.
7. Run `lightgbm_exec.py` which reads the `df_merged_train_test.pickle` file and prints `low_MAPE`, `best_params` and `best_fit_no` (see the bottom of the file). And this takes a looong time.