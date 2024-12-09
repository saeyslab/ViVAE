"""
Copyright 2024 David Novak

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from vivae import torch

class MPSDataLoader:
    """Custom data loader for compatibility with MPS backend"""

    def __init__(self, dataset, batch_size=256, shuffle=True, random_state=None):
        self.dataset = torch.tensor(dataset)
        self.n = torch.tensor(dataset.shape[0], dtype=torch.int32)

        self.batch_size = torch.tensor(batch_size, dtype=torch.int32)
        self.shuffle = shuffle

        self.len = np.ceil((self.n / self.batch_size).detach().cpu().numpy()).astype(int)

        if random_state is None:
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)

    def __iter__(self):
        idcs = torch.arange(self.n)
        start = torch.arange(0, self.n, self.batch_size)
        end = torch.minimum(start + self.batch_size, self.n)
        if self.shuffle:
            idcs = idcs[torch.randperm(self.n)]
        self.batch_indices = [torch.index_select(idcs, 0, torch.arange(i, j)) for i, j in zip(start, end, strict=False)]
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx == self.len:
            raise StopIteration
        batch = torch.index_select(self.dataset, 0, self.batch_indices[self.current_idx])
        self.current_idx += 1
        return batch, None

    def __len__(self):
        return self.len
