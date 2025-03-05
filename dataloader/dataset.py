import torch
from torch.utils.data import Dataset, DataLoader
import os
import dac
import torch.nn.functional as F  # for the integer to one-hot
import pandas as pd

class CustomDACDataset(Dataset):
    def __init__(self, data_dir, metadata_excel, transforms=None):
        """
        Args:
            data_dir (string): Directory with all the data files.
            metadata_excel (string): Path to the Excel file containing file metadata.
                                     The Excel file must have columns:
                                     'Full File Name', 'Class Name', and 'Param1'.
            transforms (callable, optional): Optional transforms to be applied on a sample.
        """
        self.data_dir = data_dir
        # Load metadata from the Excel file using Pandas
        self.metadata_df = pd.read_excel(metadata_excel)
        # Extract the list of file names from the 'Full File Name' column
        self.file_names = self.metadata_df["Full File Name"].tolist()
        # Create a mapping from file name to its metadata (as a dict)
        self.metadata_dict = self.metadata_df.set_index("Full File Name").to_dict(orient="index")
        self.transforms = transforms

        # Build the class dictionaries from the Excel file
        unique_classes = self.metadata_df["Class Name"].unique().tolist()
        unique_classes.sort()  # Sort for consistency, if desired
        self.class_name_to_int = {cls: i for i, cls in enumerate(unique_classes)}
        self.int2classname = {i: cls for cls, i in self.class_name_to_int.items()}

    def get_num_classes(self):
        """Return the number of unique classes."""
        return len(self.class_name_to_int)

    def onehot(self, class_name):
        """Return a one-hot encoded vector for the given class_name."""
        class_num = self.class_name_to_int.get(class_name, -1)
        if class_num == -1:
            print(f'class_name not found: {class_name}')
        return F.one_hot(torch.tensor(class_num), num_classes=self.get_num_classes()).to(torch.float)

    def extract_conditioning_vector(self, filename):
        """
        Retrieves the conditioning vector for a given file using metadata from the Excel file.
        Uses the 'Class Name' and 'Param1' columns.
        """
        metadata = self.metadata_dict.get(filename, None)
        if metadata is None:
            raise ValueError(f"Metadata for file {filename} not found in the Excel file")
        class_name = metadata["Class Name"]
        param_value = metadata["Param1"]
        one_hot_fvector = self.onehot(class_name)
        return torch.cat((one_hot_fvector, torch.tensor([param_value])))

    def get_class_list(self):
        """Returns a list of all unique class names."""
        return list(self.class_name_to_int.keys())

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        filename = self.file_names[idx]
        fpath = os.path.join(self.data_dir, filename)
        dacfile = dac.DACFile.load(fpath)  # Load the data file
        data = dacfile.codes

        # Assuming data is a tensor of shape [1, N, T],
        # remove the first dimension to get a tensor of shape [N, T]
        data = data.squeeze(0)

        # Input: all time steps except the last one
        input_data = data[:, :-1]
        # Target: all time steps except the first one
        target_data = data[:, 1:]
        
        condvect = self.extract_conditioning_vector(filename)
        
        # Transpose so that data has shape [T, N] for the transformer
        return input_data.transpose(0, 1), target_data.transpose(0, 1), condvect
