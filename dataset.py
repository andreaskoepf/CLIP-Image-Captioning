"""" dataset.py contains a modified version of the `NumpyMatrixReader` from criteo/autofaiss. """

from torch.utils.data import IterableDataset
from typing import Tuple, Iterator
from pathlib import Path
from abc import ABC
import numpy as np
import torch
import re
import os

def _read_numpy_header(f):
    f.seek(0, os.SEEK_END)
    file_size = f.tell()
    f.seek(0)

    first_line = f.read(
        min(file_size, 300)
    ).split(b"\n")[0]

    result = re.search(r"'shape': \(([0-9]+), ([0-9]+)(?:,\W([0-9]+))?\)", str(first_line))
    captures = result.groups()
    if captures[2] is not None:
        shape = (int(captures[0]), int(captures[1]), int(captures[2]))
    else:
        shape = (int(captures[0]), int(captures[1]))

    dtype = re.search(r"'descr': '([<(if)0-9]+)'", str(first_line)).group(1)

    end = len(first_line) + 1  # the first line content and the endline
    f.seek(0)

    return (shape, dtype, end)

class NumpyLazyNdArray(object):
    """Reads a numpy file lazily"""

    def __init__(self, f):
        self.f = f
        (self.shape, self.dtype, self.header_offset) = _read_numpy_header(f)

        self.byteperitem = np.dtype(self.dtype).itemsize
        for i in range(1, len(self.shape)):
            self.byteperitem *= self.shape[i]

        self.num_rows = self.shape[0]

    def get_rows(self, start: int, end: int) -> np.ndarray:
        length = end - start
        out_shape = (length,) + self.shape[1:]
        self.f.seek(self.header_offset + start * self.byteperitem)
        return np.frombuffer(
            self.f.read(length * self.byteperitem), dtype=self.dtype
        ).reshape(out_shape)

class NumpyMatrixReader(ABC):
    """Read a numpy file and provide its shape, row count and ndarray. Behaves as a context manager"""

    def __init__(self, file: Path):
        self.f = open(file, "rb")
    
    def close(self):
        self.f.close()

    def get_shape(self) -> Tuple[int, int]:
        shape, _, _ = _read_numpy_header(self.f)
        return shape

    def get_row_count(self) -> int:
        return self.get_shape()[0]

    def get_lazy_array(self) -> NumpyLazyNdArray:
        return NumpyLazyNdArray(self.f)


class TokenPrefixDataset(IterableDataset):
    """ IterableDataset that yields preprocessed image prefixes, text tokens and text masks. (see `preprocess_dataset.py`) """

    def __init__(self, data_path: str, batch_size: int = 5, normalize_prefix: bool = False):
        super().__init__()
        
        self.batch_size = batch_size
        self.normalize_prefix = normalize_prefix
        
        path = Path(data_path)
        prefixes_path = path / "prefixes"
        tokens_path = path / "tokens"
        
        self.prefix_files = sorted(list(prefixes_path.glob("*.npy")), key=lambda x: x.name)
        self.tokens_files = sorted(list(tokens_path.glob("*.npy")), key=lambda x: x.name)

        
        self.start_indices = [0] * len(self.prefix_files)
        self.sample_count = 0
        
        for i, file in enumerate(self.prefix_files):
            self.start_indices[i] = self.sample_count
            _reader = NumpyMatrixReader(file)
            self.sample_count += _reader.get_row_count()
            _reader.close()
    
    def __len__(self):
        return self.sample_count // self.batch_size
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        file_index = 0
        max_file_index = len(self.prefix_files)

        # `overflow_batch` is used if a batch flows over into another .npy file.
        overflow_batch = None

        while True:
            # .npy file iteration.

            if file_index >= max_file_index:
                file_index = 0

            prefix_reader = NumpyMatrixReader(self.prefix_files[file_index])
            tokens_reader = NumpyMatrixReader(self.tokens_files[file_index])

            prefix_array = prefix_reader.get_lazy_array()
            tokens_array = tokens_reader.get_lazy_array()

            sample_index = 0
            max_sample_index = prefix_array.num_rows

            while True:
                # Sample iteration inside .npy files.

                if sample_index >= max_sample_index:
                    break

                if overflow_batch is None:
                    remaining_to_add_from_overflow = self.batch_size
                else:
                    remaining_to_add_from_overflow = self.batch_size - overflow_batch[0].shape[0]
                
                add_from_reader = min(remaining_to_add_from_overflow, max_sample_index - sample_index)

                prefix_np = prefix_array.get_rows(sample_index, sample_index + add_from_reader)
                tokens_np = tokens_array.get_rows(sample_index, sample_index + add_from_reader)

                if (add_from_reader < self.batch_size) and (remaining_to_add_from_overflow == self.batch_size):
                    # File does not contain `batch_size` samples remaining, load next file...
                    overflow_batch = (prefix_np, tokens_np)
                    break

                elif (remaining_to_add_from_overflow < self.batch_size) and (overflow_batch is not None):
                    # Samples exist from previous file, concat them...
                    previous_prefix_np, previous_tokens_np = overflow_batch

                    prefix_np = np.concatenate((prefix_np, previous_prefix_np), axis=0)
                    tokens_np = np.concatenate((tokens_np, previous_tokens_np), axis=0)
                
                elif overflow_batch is not None:
                    # `overflow_batch` no longer needs to be `None`.
                    overflow_batch = None

                # Convert np array into tensors.
                tokens = torch.from_numpy(
                    np.array(tokens_np, dtype=np.int64)
                )
                prefixes = torch.from_numpy(
                    np.array(prefix_np, dtype=np.float32)
                )
            
                if self.normalize_prefix:
                    prefixes = prefixes / prefixes.norm(2, -1)
                
                sample_index += add_from_reader

                yield (tokens, prefixes)

            
            # Close file IOs to prepare next .npy files.
            prefix_reader.close()
            tokens_reader.close()

            file_index += 1
            sample_index = 0


class MultiplePrefixDataset(IterableDataset):
    """ Merges many `TokenPrefixDataset` instances together, alternating each batch. """
    
    def __init__(self, *datasets):
        super().__init__()

        self.total_samples = sum([len(dataset) for dataset in datasets])
        self.dataset_iters = [iter(dataset) for dataset in datasets]

        self.total_datasets = len(datasets)
        self.dataset_index = 0

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        while True:
            if self.dataset_index >= self.total_datasets:
                self.dataset_index = 0

            yield next(self.dataset_iters[self.dataset_index])

            self.dataset_index += 1