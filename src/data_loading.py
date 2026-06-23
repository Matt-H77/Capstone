# Data Loading Capstone Project

#imports
import numpy as np
from scipy.stats import norm



# Functions for loading initial data
# This function loads the initial data points for the specified dataset number (1-8).
# The data is stored in .npy files, which are loaded using NumPy's load function.
# The function returns the inputs and outputs as NumPy arrays.
# Parameters:
# - load_dataset: An integer (1-8) specifying which dataset to load.
def load_initial_data(load_dataset):

    if load_dataset == 1:
        function_Inputs = np.load('data/starting_data/initial_data/function_1/initial_inputs.npy')
        function_Outputs = np.load('data/starting_data/initial_data/function_1/initial_outputs.npy')

    if load_dataset == 2:
        function_Inputs = np.load('data/starting_data/initial_data/function_2/initial_inputs.npy')
        function_Outputs = np.load('data/starting_data/initial_data/function_2/initial_outputs.npy')

    if load_dataset == 3:
        function_Inputs = np.load('data/starting_data/initial_data/function_3/initial_inputs.npy')
        function_Outputs = np.load('data/starting_data/initial_data/function_3/initial_outputs.npy')

    if load_dataset == 4:
        function_Inputs = np.load('data/starting_data/initial_data/function_4/initial_inputs.npy')
        function_Outputs = np.load('data/starting_data/initial_data/function_4/initial_outputs.npy')

    if load_dataset == 5:
        function_Inputs = np.load('data/starting_data/initial_data/function_5/initial_inputs.npy')
        function_Outputs = np.load('data/starting_data/initial_data/function_5/initial_outputs.npy')

    if load_dataset == 6:
        function_Inputs = np.load('data/starting_data/initial_data/function_6/initial_inputs.npy')
        function_Outputs = np.load('data/starting_data/initial_data/function_6/initial_outputs.npy')

    if load_dataset == 7:
        function_Inputs = np.load('data/starting_data/initial_data/function_7/initial_inputs.npy')
        function_Outputs = np.load('data/starting_data/initial_data/function_7/initial_outputs.npy')

    if load_dataset == 8:
        function_Inputs = np.load('data/starting_data/initial_data/function_8/initial_inputs.npy')
        function_Outputs = np.load('data/starting_data/initial_data/function_8/initial_outputs.npy')

    return function_Inputs, function_Outputs

# This function appends a single input/output pair from text files to existing arrays of inputs and outputs.
# The input and output data are read from specified text files, and the function checks for duplicates if requested.
#
# Input data is in the form of a list of arrays in the input text file, and the output data is in the form of a list of values in the output text file.
# The id number of the function to append is specified by the position parameter, which indicates which input/output pair to select from the text files.
#
# Parameters:
# - existing_inputs: A NumPy array of shape (n_samples, n_dimensions) containing the existing input data.
# - existing_outputs: A NumPy array of shape (n_samples,) containing the existing output data.
# - input_txt_filename: A string specifying the filename of the text file containing the input data. The file should contain a list of arrays, e.g., [array([...]), array([...]), ...].
# - output_txt_filename: A string specifying the filename of the text file containing the output data. The file should contain a list of values, e.g., [np.float64(...), np.float64(...), ...].
# - position: An integer specifying which input/output pair to append from the text files.
# - prevent_duplicates: A boolean indicating whether to prevent appending an input that already exists in the existing_inputs array. If True, the function will check for duplicates and skip appending if a duplicate is found.
def append_input_and_output(
    existing_inputs,
    existing_outputs,
    input_txt_filename,
    output_txt_filename,
    position,
    prevent_duplicates=True
):
    
    # Load input file
    with open(input_txt_filename, "r") as f:
        input_text = f.read()

    input_points = eval(
        input_text,
        {"__builtins__": None, "array": np.array},
        {}
    )

    # Load output file
    with open(output_txt_filename, "r") as f:
        output_text = f.read()

    output_values = eval(
        output_text,
        {"__builtins__": None, "np": np},
        {}
    )

    output_values = np.asarray(
        output_values,
        dtype=float
    )

    # Validate position
    if position < 0 or position >= len(input_points):
        raise IndexError(
            f"position must be between 0 and "
            f"{len(input_points)-1}"
        )

    if position >= len(output_values):
        raise IndexError(
            f"output file only contains "
            f"{len(output_values)} values"
        )

    # Select input/output pair
    selected_input = np.asarray(
        input_points[position],
        dtype=float
    )

    selected_output = float(
        output_values[position]
    )

    # Validate dimensionality
    expected_dimensions = existing_inputs.shape[1]

    if selected_input.shape[0] != expected_dimensions:
        raise ValueError(
            f"Selected input has "
            f"{selected_input.shape[0]} dimensions "
            f"but existing_inputs expects "
            f"{expected_dimensions}"
        )

    # Optional duplicate protection
    if prevent_duplicates:

        already_exists = np.any(
            np.all(
                np.isclose(
                    existing_inputs,
                    selected_input,
                    atol=1e-12
                ),
                axis=1
            )
        )

        if already_exists:
            print(
                "Input already exists; "
                "not appending again."
            )
            return (existing_inputs, existing_outputs)

    # Append the selected input and output to the existing arrays.
    existing_inputs = np.vstack([existing_inputs, selected_input])
    existing_outputs = np.append(existing_outputs, selected_output)
    print(f"Appended position {position} which is data from {input_txt_filename} for Function {position+1}.")
    return (existing_inputs, existing_outputs)
