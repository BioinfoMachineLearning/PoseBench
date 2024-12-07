from beartype.typing import Any, List
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    """A `Dataset` that combines two datasets."""

    def __init__(self, dataset1: Dataset, dataset2: Dataset):
        """Initialize a `CombinedDataset`."""
        super().__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        """Return the length of the combined dataset."""
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx: int):
        """Return the example at the given index."""
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]

    def add_examples(self, new_example_list: List[Any]):
        """Add new examples to the first dataset."""
        self.dataset1.add_examples(new_example_list)
