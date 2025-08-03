from torch.utils.data import Dataset
import torchvision.transforms as T
from datasets import load_dataset


class CC3MDataset(Dataset):

    def __init__(self, split='validation', tokenizer=None):
        train_data_files = [f"cc3m-train-{str(i).zfill(4)}.tar" for i in range(25)]
        val_data_files = [f"cc3m-validation-{str(i).zfill(4)}.tar" for i in range(15)]
        data_files = {"train": train_data_files, "validation": val_data_files}
        self.dataset = load_dataset('pixparse/cc3m-wds', split=split, data_files=data_files)      
        #self.dataset = self.dataset.with_format("torch")
        self.tokenizer = tokenizer
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.transform(item['jpg'])
        input_ids = self.tokenizer(item['txt'].strip(), return_tensors='pt', padding='max_length', truncation=True, max_length=77)
        input_ids = input_ids['input_ids'].squeeze(0)
        return image, input_ids