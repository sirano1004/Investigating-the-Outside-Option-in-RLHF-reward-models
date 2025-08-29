
from datasets import load_dataset


def get_data(split:str,start_row: int, end_row: int):
    """
    Downloads the hh-rlhf dataset, selects a specific range of rows,
    truncates the final assistant response from the prompt, and saves it to a CSV file.

    Args:
        start_row (int): The starting row number to select from the dataset.
        end_row (int): The ending row number to select from the dataset.
        output_filename (str): The name of the CSV file to save the results to.
    """
    # Define the dataset and the slice
    dataset_name = "stanfordnlp/imdb"
    row_slice = f"{split}[{start_row}:{end_row}]"
    
    print(f"Loading rows {start_row} to {end_row} from '{dataset_name}'...")
    
    try:
        # Load the specified slice of the dataset
        dataset = load_dataset(dataset_name, split=row_slice)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    return dataset['text']