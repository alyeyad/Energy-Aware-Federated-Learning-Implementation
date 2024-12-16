import numpy as np
import time
import torch

# def low_precision_quantizer_with_sign(matrix, num_levels):
#     """
#     Implements a low-precision quantizer for 2D matrices, handling sign.
#
#     Parameters:
#     matrix (np.array or torch.Tensor): The 2D matrix to be quantized.
#     num_levels (int): The number of quantization levels.
#
#     Returns:
#     np.array or torch.Tensor: Quantized matrix.
#     dict: Information about bandwidth savings.
#     str: Quantized matrix represented as bits.
#     dict: Mapping of levels to their corresponding values.
#     str: Sign vector.
#     """
#     # Convert the norm to a Python float
#     norm = float(matrix.norm().item()) if hasattr(matrix, "norm") else np.linalg.norm(matrix)
#     quantized_matrix = matrix.clone() if hasattr(matrix, "clone") else np.zeros_like(matrix)
#
#     # Define the range of levels in [0, 1]
#     step_size = 1 / (num_levels - 1)  # Step size for quantization levels
#     level_mapping = {i: i * step_size * norm for i in range(num_levels)}  # Map levels to quantized values
#
#     # Initialize sign vector
#     sign_vector = ''
#
#     # Quantize each element
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             value = matrix[i, j]
#             print(matrix.shape)
#             sign = 1 if value >= 0 else -1
#             sign_vector += '1' if sign >= 0 else '0'  # Store sign as a bit
#
#             # Normalize value and find the closest level
#             normalized_value = abs(value).item() / norm if hasattr(value, "item") else abs(value) / norm
#             quantized_level = round(normalized_value / step_size)  # Ensure value is a Python scalar
#             quantized_matrix[i, j] = level_mapping[quantized_level] * sign  # Apply quantized value with sign
#
#     # Represent quantized matrix as a bit vector
#     quantized_bits_per_param = int(np.ceil(np.log2(num_levels)))
#     bit_vector = ''.join([
#         bin(int(round(abs(matrix[i, j].item() if hasattr(matrix[i, j], "item") else matrix[i, j]) / norm / step_size)))[2:].zfill(quantized_bits_per_param)
#         for i in range(matrix.shape[0]) for j in range(matrix.shape[1])
#     ])
#
#     # Bandwidth savings calculation
#     num_parameters = matrix.numel() if hasattr(matrix, "numel") else matrix.size  # Total elements
#     full_precision_bits = 32  # Bits per parameter in full precision
#     quantized_bits_per_param = int(np.ceil(np.log2(num_levels))) + 1  # Add 1 bit for the sign
#
#     full_precision_data_size = num_parameters * full_precision_bits / 8 / 1_000_000  # MB
#     quantized_data_size = (num_parameters * quantized_bits_per_param) / 8 / 1_000_000  # MB
#     bandwidth_savings = (full_precision_data_size - quantized_data_size) / full_precision_data_size * 100
#
#     savings_info = {
#         "full_precision_data_size_MB": full_precision_data_size,
#         "quantized_data_size_MB": quantized_data_size,
#         "bandwidth_savings_percent": bandwidth_savings
#     }
#
#     return quantized_matrix, savings_info, bit_vector, level_mapping, sign_vector
def low_precision_quantizer_4d(matrix, num_levels):
    """
    Implements a low-precision quantizer for 4D tensors, handling sign.

    Parameters:
    matrix (np.array or torch.Tensor): The 4D tensor to be quantized.
    num_levels (int): The number of quantization levels.

    Returns:
    np.array or torch.Tensor: Quantized tensor.
    dict: Information about bandwidth savings.
    str: Quantized tensor represented as bits.
    dict: Mapping of levels to their corresponding values.
    str: Sign vector.
    """
    # Check if the input is a PyTorch tensor
    is_tensor = hasattr(matrix, "clone")

    # Calculate the norm
    norm = float(matrix.norm().item()) if is_tensor else np.linalg.norm(matrix)
    quantized_matrix = matrix.clone() if is_tensor else np.zeros_like(matrix)

    # Define the range of levels in [0, 1]
    step_size = 1 / (num_levels - 1)  # Step size for quantization levels
    level_mapping = {i: i * step_size * norm for i in range(num_levels)}  # Map levels to quantized values

    # Initialize sign vector
    sign_vector = ''

    # Quantize each element
    for idx in range(matrix.numel() if is_tensor else matrix.size):
        # Get the flattened index and corresponding multi-dimensional index
        multi_idx = np.unravel_index(idx, matrix.shape)
        value = matrix[multi_idx].item() if is_tensor else matrix[multi_idx]

        # Determine the sign
        sign = 1 if value >= 0 else -1
        sign_vector += '1' if sign >= 0 else '0'  # Store sign as a bit

        # Normalize value and find the closest level
        normalized_value = abs(value) / norm if norm != 0 else 0
        quantized_level = round(normalized_value / step_size)  # Find the closest level
        quantized_value = level_mapping[quantized_level] * sign  # Apply quantized value with sign

        if is_tensor:
            quantized_matrix[multi_idx] = quantized_value  # Assign in tensor
        else:
            quantized_matrix[multi_idx] = quantized_value  # Assign in numpy array

    # Represent quantized tensor as a bit vector
    quantized_bits_per_param = int(np.ceil(np.log2(num_levels)))
    bit_vector = ''.join([
        bin(int(round(abs(matrix[multi_idx].item() if is_tensor else matrix[multi_idx]) / norm / step_size)))[2:].zfill(
            quantized_bits_per_param)
        for idx in range(matrix.numel() if is_tensor else matrix.size)
        for multi_idx in [np.unravel_index(idx, matrix.shape)]
    ])

    # Bandwidth savings calculation
    num_parameters = matrix.numel() if is_tensor else matrix.size  # Total elements
    full_precision_bits = 32  # Bits per parameter in full precision
    quantized_bits_per_param = int(np.ceil(np.log2(num_levels))) + 1  # Add 1 bit for the sign

    full_precision_data_size = num_parameters * full_precision_bits / 8 / 1_000_000  # MB
    quantized_data_size = (num_parameters * quantized_bits_per_param) / 8 / 1_000_000  # MB
    bandwidth_savings = (full_precision_data_size - quantized_data_size) / full_precision_data_size * 100

    savings_info = {
        "full_precision_data_size_MB": full_precision_data_size,
        "quantized_data_size_MB": quantized_data_size,
        "bandwidth_savings_percent": bandwidth_savings
    }

    return quantized_matrix, savings_info, bit_vector, level_mapping, sign_vector


# def bit_vector_to_matrix_with_sign(bit_vector, sign_vector, level_mapping, matrix_shape, num_levels):
#     """
#     Converts a bit vector and sign vector back to the quantized 2D matrix.
#
#     Parameters:
#     bit_vector (str): Bit representation of the quantized matrix.
#     sign_vector (str): Sign vector of the quantized matrix.
#     level_mapping (dict): Mapping of quantization levels to their corresponding values.
#     matrix_shape (tuple): Shape of the original 2D matrix (rows, columns).
#     num_levels (int): Number of quantization levels.
#
#     Returns:
#     np.array: Reconstructed quantized 2D matrix.
#     """
#     quantized_bits_per_param = int(np.ceil(np.log2(num_levels)))  # Bits per parameter
#
#     # Split the bit vector into chunks for each parameter
#     chunks = [bit_vector[i:i + quantized_bits_per_param] for i in range(0, len(bit_vector), quantized_bits_per_param)]
#
#     # Decode each chunk into the corresponding quantized level
#     reconstructed_matrix = np.zeros(matrix_shape)
#     for idx, chunk in enumerate(chunks):
#         level = int(chunk, 2)  # Convert binary to integer level
#         value = level_mapping[level]  # Map level to quantized value
#         sign = 1 if sign_vector[idx] == '1' else -1  # Get sign from sign vector
#         row, col = divmod(idx, matrix_shape[1])  # Map flat index to 2D indices
#         reconstructed_matrix[row, col] = value * sign  # Apply the sign
#
#     return reconstructed_matrix


# def simulate_delay(quantized_matrix, savings_info, num_levels, bandwidth_mbps, sleep_for_delay=False):
#     """
#     Simulates the delay caused by transmitting quantized and unquantized 2D matrices.
#
#     Parameters:
#     matrix (np.array): The original 2D matrix.
#     num_levels (int): Number of quantization levels.
#     bandwidth_mbps (float): Bandwidth in megabits per second.
#     sleep_for_delay (bool): Whether to simulate delay using sleep.
#
#     Returns:
#     dict: Transmission delays for quantized and unquantized matrices.
#     """
#     # Quantize the matrix
#
#     num_parameters = quantized_matrix.size()[0] * quantized_matrix.size()[1]  # Total number of parameters in the 2D matrix
#     print(num_parameters)
#     full_precision_bits = 32  # Bits per parameter in full precision
#     quantized_bits_per_param = int(np.ceil(np.log2(num_levels))) + 1  # Add 1 bit for the sign
#
#     # Calculate transmission time for full precision
#     full_precision_data_size_bits = num_parameters * full_precision_bits  # Size in bits
#     full_precision_time = full_precision_data_size_bits / (bandwidth_mbps * 1_000_000)  # Time in seconds
#
#     # Calculate transmission time for quantized precision
#     quantized_data_size_bits = num_parameters * quantized_bits_per_param  # Size in bits
#     quantized_time = quantized_data_size_bits / (bandwidth_mbps * 1_000_000)  # Time in seconds
#
#     if sleep_for_delay:
#         print("Simulating delay for full-precision transmission...")
#         time.sleep(full_precision_time)
#         print("Full-precision transmission completed.")
#
#         print("Simulating delay for quantized transmission...")
#         time.sleep(quantized_time)
#         print("Quantized transmission completed.")
#
#     return {
#         "full_precision_time_seconds": full_precision_time,
#         "quantized_time_seconds": quantized_time,
#         "time_savings_percent": (full_precision_time - quantized_time) / full_precision_time * 100
#     }
def simulate_delay_4d(quantized_matrix, num_levels, bandwidth_mbps, sleep_for_delay=False):
    """
    Simulates the delay caused by transmitting quantized and unquantized 4D tensors.

    Parameters:
    matrix (np.array or torch.Tensor): The original 4D tensor.
    num_levels (int): Number of quantization levels.
    bandwidth_mbps (float): Bandwidth in megabits per second.
    sleep_for_delay (bool): Whether to simulate delay using sleep.

    Returns:
    dict: Transmission delays for quantized and unquantized tensors.
    """
    # Quantize the tensor
    # quantized_matrix, savings_info, _, _, _ = low_precision_quantizer_4d(matrix, num_levels)

    num_parameters = quantized_matrix.numel() if hasattr(quantized_matrix, "numel") else quantized_matrix.size  # Total number of parameters
    full_precision_bits = 32  # Bits per parameter in full precision
    quantized_bits_per_param = int(np.ceil(np.log2(num_levels))) + 1  # Add 1 bit for the sign

    # Calculate transmission time for full precision
    full_precision_data_size_bits = num_parameters * full_precision_bits  # Size in bits
    full_precision_time = full_precision_data_size_bits / (bandwidth_mbps * 1_000_000)  # Time in seconds

    # Calculate transmission time for quantized precision
    quantized_data_size_bits = num_parameters * quantized_bits_per_param  # Size in bits
    quantized_time = quantized_data_size_bits / (bandwidth_mbps * 1_000_000)  # Time in seconds

    if sleep_for_delay:
        print("Simulating delay for full-precision transmission...")
        time.sleep(full_precision_time)
        print("Full-precision transmission completed.")

        print("Simulating delay for quantized transmission...")
        time.sleep(quantized_time)
        print("Quantized transmission completed.")

    return {
        "full_precision_time_seconds": full_precision_time,
        "quantized_time_seconds": quantized_time,
        "time_savings_percent": (full_precision_time - quantized_time) / full_precision_time * 100
    }

def bit_vector_to_tensor_with_sign_4d(bit_vector, sign_vector, level_mapping, tensor_shape, num_levels):
    """
    Converts a bit vector and sign vector back to the quantized 4D tensor.

    Parameters:
    bit_vector (str): Bit representation of the quantized tensor.
    sign_vector (str): Sign vector of the quantized tensor.
    level_mapping (dict): Mapping of quantization levels to their corresponding values.
    tensor_shape (tuple): Shape of the original 4D tensor (dimensions: [dim1, dim2, dim3, dim4]).
    num_levels (int): Number of quantization levels.

    Returns:
    np.array or torch.Tensor: Reconstructed quantized 4D tensor.
    """
    quantized_bits_per_param = int(np.ceil(np.log2(num_levels)))  # Bits per parameter

    # Split the bit vector into chunks for each parameter
    chunks = [bit_vector[i:i+quantized_bits_per_param] for i in range(0, len(bit_vector), quantized_bits_per_param)]

    # Decode each chunk into the corresponding quantized level
    reconstructed_tensor = np.zeros(tensor_shape)  # Default to Numpy array
    is_tensor = False

    if hasattr(tensor_shape, "__torch_function__"):  # Check if PyTorch is used
        is_tensor = True
        reconstructed_tensor = torch.zeros(tensor_shape)

    for idx, chunk in enumerate(chunks):
        level = int(chunk, 2)  # Convert binary to integer level
        value = level_mapping[level]  # Map level to quantized value
        sign = 1 if sign_vector[idx] == '1' else -1  # Get sign from sign vector
        multi_idx = np.unravel_index(idx, tensor_shape)  # Convert flat index to 4D index
        if is_tensor:
            reconstructed_tensor[multi_idx] = value * sign
        else:
            reconstructed_tensor[multi_idx] = value * sign

    return reconstructed_tensor



if __name__ == "__main__":
    # Define a 4D tensor
    original_tensor = torch.randn(6, 3, 5, 5)  # Random 4D tensor
    num_levels = 32  # Number of quantization levels

    # Quantize the 4D tensor
    quantized_tensor, savings_info, quantized_tensor_bits, level_mapping, sign_vector = low_precision_quantizer_4d(
        original_tensor, num_levels)
    bandwidth_mbps = 10  # Bandwidth in megabits per second

    # Simulate delay
    delay_info = simulate_delay_4d(quantized_tensor, num_levels, bandwidth_mbps, sleep_for_delay=False)
    # Reconstruct the 4D tensor from the bit vector and sign vector
    reconstructed_tensor = bit_vector_to_tensor_with_sign_4d(quantized_tensor_bits, sign_vector, level_mapping,
                                                             original_tensor.shape, num_levels)


    print("Original Tensor:")
    print(original_tensor)
    print("\nQuantized Tensor:")
    print(quantized_tensor)
    print("\nBit Vector:")
    print(quantized_tensor_bits[:100] + " ...")  # Truncate for readability
    print("\nSign Vector:")
    print(sign_vector[:100] + " ...")  # Truncate for readability
    print("\nLevel Mapping:")
    print(level_mapping)
    print("\nBandwidth Savings:")
    print(savings_info)
    print("Transmission Delay Info:")
    print(delay_info)

