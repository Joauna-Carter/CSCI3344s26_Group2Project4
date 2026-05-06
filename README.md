# CSCI3344s26_Group2Project4
main.py - MLP with 2 hidden layers  

main2.py - MLP with 3 hidden layers  

jcmain.py - 2-layer MLP model with preprocessing, training, evaluation, accuracy graphing, and result saving for each run 

jcvisual.py - Used for data exploration and analysis. Displays histograms, statistical summaries, missing value counts, and compares diabetes vs non-diabetes groups.

results_logger.py - Handles saving results from each model run. Stores accuracy graphs, per-epoch results, and a summary of each run in CSV format inside the model_results folder.

tune_hyperparams.py - attempt at finding better hidden layer sizes/epochs  

More Detailed for All Files Below:

main.py - MLP with 2 hidden layers. This was the original baseline model used to test performance and compare against other configurations.

main2.py - MLP with 3 hidden layers. This model was tested to see if adding another layer would improve performance, but results were slightly worse than the baseline.

jcmain.py - 2-layer MLP model with preprocessing, training, evaluation, accuracy graphing, and result saving for each run

jcvisual.py - Used for data exploration and analysis. Displays histograms, statistical summaries, missing value counts, and compares diabetes vs non-diabetes groups.

results_logger.py - Handles saving results from each model run. Stores accuracy graphs, per-epoch results, and a summary of each run in CSV format inside the model_results folder.

tune_hyperparams.py - Runs random search to test different hidden layer sizes and epoch values. Used to find better model configurations by averaging results over multiple runs.

model_results/ - Folder containing all saved experiment outputs. Each run has its own timestamped folder with graphs and CSV files. Also includes a general_results.csv file summarizing all runs.

general_results.csv - Summary file that tracks all model runs, including training accuracy, validation accuracy, test accuracy, and loss values.

run_*/ folders - Each folder represents one model run. Contains:
  - accuracy graph image
  - per-epoch results CSV

before_stats_summary.csv - Statistical summary of the dataset before preprocessing (includes invalid 0 values).

after_stats_summary.csv - Statistical summary after preprocessing (missing values handled and data standardized).

diabetes.csv - The dataset used for training and testing the model.

environment.yml - Lists required libraries and versions to recreate the project environment.

how_to_download_the_dataset.txt - Instructions on how to obtain the dataset if it is not included.

.gitignore - Specifies which files/folders (like __pycache__) should not be tracked by Git.