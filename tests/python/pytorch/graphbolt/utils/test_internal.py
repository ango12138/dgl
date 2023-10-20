import os
import tempfile

import dgl.graphbolt.utils as utils
import numpy as np
import pytest
import torch


def test_read_torch_data():
    with tempfile.TemporaryDirectory() as test_dir:
        save_tensor = torch.tensor([[1, 2, 4], [2, 5, 3]])
        file_name = os.path.join(test_dir, "save_tensor.pt")
        torch.save(save_tensor, file_name)
        read_tensor = utils.internal._read_torch_data(file_name)
        assert torch.equal(save_tensor, read_tensor)
        save_tensor = read_tensor = None


@pytest.mark.parametrize("in_memory", [True, False])
def test_read_numpy_data(in_memory):
    with tempfile.TemporaryDirectory() as test_dir:
        save_numpy = np.array([[1, 2, 4], [2, 5, 3]])
        file_name = os.path.join(test_dir, "save_numpy.npy")
        np.save(file_name, save_numpy)
        read_tensor = utils.internal._read_numpy_data(file_name, in_memory)
        assert torch.equal(torch.from_numpy(save_numpy), read_tensor)
        save_numpy = read_tensor = None


@pytest.mark.parametrize("fmt", ["torch", "numpy"])
def test_read_data(fmt):
    with tempfile.TemporaryDirectory() as test_dir:
        data = np.array([[1, 2, 4], [2, 5, 3]])
        type_name = "pt" if fmt == "torch" else "npy"
        file_name = os.path.join(test_dir, f"save_data.{type_name}")
        if fmt == "numpy":
            np.save(file_name, data)
        elif fmt == "torch":
            torch.save(torch.from_numpy(data), file_name)
        read_tensor = utils.read_data(file_name, fmt)
        assert torch.equal(torch.from_numpy(data), read_tensor)


@pytest.mark.parametrize(
    "data_fmt, save_fmt, contiguous",
    [
        ("torch", "torch", True),
        ("torch", "torch", False),
        ("torch", "numpy", True),
        ("torch", "numpy", False),
        ("numpy", "torch", True),
        ("numpy", "torch", False),
        ("numpy", "numpy", True),
        ("numpy", "numpy", False),
    ],
)
def test_save_data(data_fmt, save_fmt, contiguous):
    with tempfile.TemporaryDirectory() as test_dir:
        data = np.array([[1, 2, 4], [2, 5, 3]])
        if not contiguous:
            data = np.asfortranarray(data)
        tensor_data = torch.from_numpy(data)
        type_name = "pt" if save_fmt == "torch" else "npy"
        save_file_name = os.path.join(test_dir, f"save_data.{type_name}")
        # Step1. Save the data.
        if data_fmt == "torch":
            utils.save_data(tensor_data, save_file_name, save_fmt)
        elif data_fmt == "numpy":
            utils.save_data(data, save_file_name, save_fmt)

        # Step2. Load the data.
        if save_fmt == "torch":
            loaded_data = torch.load(save_file_name)
            assert loaded_data.is_contiguous()
            assert torch.equal(tensor_data, loaded_data)
        elif save_fmt == "numpy":
            loaded_data = np.load(save_file_name)
            # Checks if the loaded data is C-contiguous.
            assert loaded_data.flags["C_CONTIGUOUS"]
            assert np.array_equal(tensor_data.numpy(), loaded_data)

        data = tensor_data = loaded_data = None


@pytest.mark.parametrize("fmt", ["torch", "numpy"])
def test_get_npy_dim(fmt):
    with tempfile.TemporaryDirectory() as test_dir:
        data = np.array([[1, 2, 4], [2, 5, 3]])
        type_name = "pt" if fmt == "torch" else "npy"
        file_name = os.path.join(test_dir, f"save_data.{type_name}")
        if fmt == "numpy":
            np.save(file_name, data)
            assert utils.get_npy_dim(file_name) == 2
        elif fmt == "torch":
            torch.save(torch.from_numpy(data), file_name)
            with pytest.raises(ValueError):
                utils.get_npy_dim(file_name)
        data = None


@pytest.mark.parametrize("data_fmt", ["numpy", "torch"])
@pytest.mark.parametrize("save_fmt", ["numpy", "torch"])
@pytest.mark.parametrize("is_feature", [True, False])
def test_copy_or_convert_data(data_fmt, save_fmt, is_feature):
    with tempfile.TemporaryDirectory() as test_dir:
        data = np.arange(10)
        tensor_data = torch.from_numpy(data)
        in_type_name = "npy" if data_fmt == "numpy" else "pt"
        input_path = os.path.join(test_dir, f"data.{in_type_name}")
        out_type_name = "npy" if save_fmt == "numpy" else "pt"
        output_path = os.path.join(test_dir, f"out_data.{out_type_name}")
        if data_fmt == "numpy":
            np.save(input_path, data)
        else:
            torch.save(tensor_data, input_path)
        utils.copy_or_convert_data(
            input_path, output_path, data_fmt, save_fmt, is_feature=is_feature
        )
        if save_fmt == "numpy":
            out_data = np.load(output_path)
        else:
            out_data = torch.load(output_path)
        if is_feature:
            data = data.reshape(-1, 1)
            tensor_data = tensor_data.reshape(-1, 1)
        if save_fmt == "numpy":
            assert (data == out_data).all()
        else:
            assert torch.equal(tensor_data, out_data)
        data = None
        tensor_data = None
        out_data = None
