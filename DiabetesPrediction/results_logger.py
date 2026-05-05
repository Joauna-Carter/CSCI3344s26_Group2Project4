import os
import time
import pandas as pd


def save_results(history, accuracy, epochs, batch_size, plt):
    # create main folder
    os.makedirs("model_results", exist_ok=True)

    # timestamp
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # create folder for this run
    run_folder = f"model_results/run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)

    
    # Save graph (WITH timestamp)
   
    graph_filename = f"{run_folder}/accuracy_graph_{timestamp}.png"
    plt.savefig(graph_filename)

    
    # Save epoch results (WITH timestamp)
    
    epoch_filename = f"{run_folder}/epoch_results_{timestamp}.csv"

    epoch_df = pd.DataFrame(history.history)
    epoch_df.index = epoch_df.index + 1
    epoch_df.index.name = "Epoch"
    epoch_df.to_csv(epoch_filename)

   
    # Summary file (one row per run)
    
    general_csv = "model_results/general_results.csv"

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    new_row = {
        "RunTime": timestamp,
        "Epochs": epochs,
        "BatchSize": batch_size,
        "TrainAccuracy": round(train_acc, 4),
        "ValidationAccuracy": round(val_acc, 4),
        "TestAccuracy": round(accuracy, 4),
        "RunFolder": run_folder,
        "GraphFile": graph_filename,
        "EpochFile": epoch_filename
    }

    # append or create
    if os.path.exists(general_csv):
        df_results = pd.read_csv(general_csv)
        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_results = pd.DataFrame([new_row])

    df_results.to_csv(general_csv, index=False)