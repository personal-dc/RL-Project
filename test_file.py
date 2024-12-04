import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import math



buffer_capacity = 10
batch_size = 4

batches_torch = BatchSampler(SubsetRandomSampler(range(buffer_capacity)), batch_size, False)
print([batch for batch in batches_torch])

batches =  np.array_split(np.random.permutation(np.arange(buffer_capacity)), math.ceil(buffer_capacity / batch_size))
print(batches)

x = np.array([1, 2, 3, 4], dtype=np.float64)
print(x)