from networks.vit_seg_modeling import CONFIGS, VisionTransformer
from networks.segcd_net import SegCDNet
from networks.segm import get_segmenter
from torchinfo import summary
import torch
import os
from torch import Tensor
from typing import Any, Tuple, List
from PIL import Image
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm
from torchvision.transforms import ToTensor
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT, TRAIN_DATALOADERS, OptimizerLRScheduler
from sklearn.model_selection import KFold
from torch import optim
import random
from sklearn.metrics import confusion_matrix, classification_report

def read_images(image_paths, crop_size = 256) -> List[Image.Image]:
    images = []
    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        
        width, height = image.size
        
        for i in range(width // crop_size):
            for j in range(height // crop_size):
                images.append(image.crop((i * crop_size, j * crop_size, (i + 1) * crop_size, (j + 1) * crop_size)))
    return images
    

class SecondDataset(Dataset):
    def __init__(self, images: List[Image.Image], labels: List[Image.Image]) -> None:
        super().__init__()
        
        self.images = images
        self.labels = labels
        
        colormap = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
        # classes = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']
        colormap2label = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        self.colormap2label = colormap2label
        
        
    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        image: Image.Image = self.images[index]
        rgb_label: Image.Image = self.labels[index]
        label = np.array(rgb_label)
        
        label = self.colormap2label[(label[:, :, 0] * 256 + label[:, :, 1]) * 256 + label[:, :, 2]]
        
        label = label * (label < len(rgb_label.getcolors()))
        
        image_tensor = ToTensor()(image)
        label_tensor = ToTensor()(label)
        
        return image_tensor, label_tensor.squeeze().long()
    
    def __len__(self):
        return len(self.images)


class KFlodDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, mode = ['train', 'val'], image_size = 256, image_suffix = 'im1', label_suffix = 'label1', batch_size=12, n_splits=5) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.image_size = image_size
        self.data_dir = data_dir
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        
        
        train_image_list = [os.path.join(self.data_dir, mode[0], self.image_suffix, image_name) for image_name in os.listdir(os.path.join(self.data_dir, mode[0], self.image_suffix))]
        train_label_list = [os.path.join(self.data_dir, mode[0], self.label_suffix, image_name) for image_name in os.listdir(os.path.join(self.data_dir, mode[0], self.label_suffix))]
        val_image_list = [os.path.join(self.data_dir, mode[1], self.image_suffix, image_name) for image_name in os.listdir(os.path.join(self.data_dir, mode[1], self.image_suffix))]
        val_label_list = [os.path.join(self.data_dir, mode[1], self.label_suffix, image_name) for image_name in os.listdir(os.path.join(self.data_dir, mode[1], self.label_suffix))]
        train_images = read_images(train_image_list, self.image_size)
        train_labels = read_images(train_label_list, self.image_size)
        val_images = read_images(val_image_list, self.image_size)
        val_labels = read_images(val_label_list, self.image_size)
        self.train_dataset = SecondDataset(images=train_images, labels=train_labels)
        self.val_dataset = SecondDataset(images=val_images, labels=val_labels)
        if (self.n_splits > 0):
            self.kfold = KFold(n_splits=self.n_splits)
            total_images = []
            total_labels = []
            total_images.extend(train_images)
            total_images.extend(val_images)
            total_labels.extend(train_labels)
            total_labels.extend(val_labels)
            self.dataset = SecondDataset(images=total_images, labels=total_labels)
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)
            
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.train_loader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.val_loader
    
    def train_fold(self, train_indices):
        assert self.n_splits > 0, "in kfold mode, the n_split param must bigger than 0, or use normal dataloader"
        train_dataset = Subset(self.dataset, train_indices)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)
    
    def val_fold(self, val_indices):
        assert self.n_splits > 0, "in kfold mode, the n_split param must bigger than 0, or use normal dataloader"
        val_dataset = Subset(self.dataset, val_indices)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=8, shuffle=False)


class LitVisionTransformer(pl.LightningModule):
    def __init__(self, config=CONFIGS["R50-ViT-B_16"], base_lr=1e-2) -> None:
        super().__init__()
        self.net = VisionTransformer(config)
        self.criterion_ce = CrossEntropyLoss()
        self.criterion_dice = DiceLoss(config["n_classes"])
        self.config = config
        self.base_lr = base_lr
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.SGD(self.net.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=0.0001)
    
    def forward(self, it: Tensor) -> Tensor:
        return self.net(it)
    
    def backward(self, loss: Tensor) -> None:
        loss.backward()
    
    def training_step(self, batch) -> STEP_OUTPUT:
        it, lt = batch
        it: Tensor
        lt: Tensor
        ot: Tensor = self(it)
        loss_ce: Tensor = self.criterion_ce(ot, lt[:].long())
        loss_dice: Tensor = self.criterion_dice(ot, lt, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('train/loss', loss.item(), sync_dist=True)
        return loss
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        it, lt = batch
        it: Tensor
        lt: Tensor
        ot: Tensor = self(it)
        loss_ce: Tensor = self.criterion_ce(ot, lt[:].long())
        loss_dice: Tensor = self.criterion_dice(ot, lt, softmax=True)
        loss = 0.5 * loss_ce + 0.5 * loss_dice
        self.log('val/loss', loss.item(), sync_dist=True)
        tp = np.zeros(self.config["n_classes"] - 1)
        fp = np.zeros(self.config["n_classes"] - 1)
        fn = np.zeros(self.config["n_classes"] - 1)
        out = torch.argmax(torch.softmax(ot, dim=1), dim=1).squeeze(0)
        prediction = out.detach().cpu().numpy()
        label = lt.detach().cpu().numpy()
        for cat in range(self.config["n_classes"] - 1):
            tp[cat] += ((prediction == cat) & (label == cat) & (label < self.config["n_classes"] - 1)).sum()
            fp[cat] += ((prediction == cat) & (label != cat) & (label < self.config["n_classes"] - 1)).sum()
            fn[cat] += ((prediction != cat) & (label == cat) & (label < self.config["n_classes"] - 1)).sum()
        np.seterr(divide='ignore', invalid='ignore')
        iou = np.divide(tp, tp + fp + fn)
        miou = iou.mean()
        self.log('val/miou', miou, sync_dist=True)
        
        return loss


class LitSeg(pl.LightningModule):
    def __init__(self, classes=['unchanged', 'water', 'ground', 'low_vegetation', 'tree', 'building', 'sports_field'], base_lr=1e-2) -> None:
        super().__init__()
        num_classes = len(classes)
        self.num_classes = num_classes
        self.classes = classes
        
        self.net = SegCDNet(num_classes=num_classes)
        # self.net = get_segmenter()

        self.criterion = CrossEntropyLoss()
        self.base_lr = base_lr
        self.train_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.val_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
        self.val_cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=self.device)
        self.num_of_trian_batches = 0
        self.num_of_val_batches = 0
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.SGD(self.net.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=0.0001)
    
    def forward(self, it: Tensor) -> Tensor:
        return self.net(it)
    
    def backward(self, loss: Tensor) -> None:
        loss.backward()
    
    def on_train_epoch_start(self) -> None:
        self.num_of_trian_batches = 0
        self.train_loss = self.train_loss.zero_().to(self.device)
    
    def training_step(self, batch) -> STEP_OUTPUT:
        it, lt = batch
        it: Tensor
        lt: Tensor
        ot: Tensor = self(it)
        loss: Tensor = self.criterion(ot, lt)
        self.train_loss = self.train_loss + loss
        self.num_of_trian_batches = self.num_of_trian_batches + 1
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log('loss/train', self.train_loss / self.num_of_trian_batches, sync_dist=True)
        
    def on_validation_epoch_start(self) -> None:
        self.num_of_val_batches = 0
        self.val_loss = self.val_loss.zero_().to(self.device)
        self.val_cm = self.val_cm.zero_().to(self.device)
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        it, lt = batch
        it: Tensor
        lt: Tensor
        ot: Tensor = self(it)
        loss: Tensor = self.criterion(ot, lt)
        self.val_loss = self.val_loss + loss
        self.num_of_val_batches = self.num_of_val_batches + 1
        
        pt = torch.argmax(ot, 1)
        
        cm = torch.tensor([[((lt == i) & (pt == j)).sum() for j in range(self.num_classes)] 
                           for i in range(self.num_classes)], dtype=torch.long, device=self.device)
        
        self.val_cm = self.val_cm + cm
        
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('loss/val', self.val_loss / self.num_of_val_batches, sync_dist=True)
        self.log('val/acc', self.val_cm.trace() / self.val_cm.sum(), sync_dist=True)
        mp = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        mr = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        mf1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        miou = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for i in range(self.num_classes):
            precision = self.val_cm[i, i] / (self.val_cm[i, :].sum() + 1e-10)
            recall = self.val_cm[i, i] / (self.val_cm[:, i].sum() + 1e-10)
            f1_score = 2 * precision * recall / (precision + recall + 1e-10)
            iou = self.val_cm[i, i] / (self.val_cm[i, :].sum() + self.val_cm[:, i].sum() - 2 * self.val_cm[i, i] + 1e-10)
            mp = mp + precision
            mr = mr + recall
            mf1 = mf1 + f1_score
            miou = miou + iou
            self.log(f'metrics/precision_{self.classes[i]}', precision, sync_dist=True)
            self.log(f'metrics/recall_{self.classes[i]}', recall, sync_dist=True)
            self.log(f'metrics/f1_score_{self.classes[i]}', f1_score, sync_dist=True)
            self.log(f'metrics/iou_{self.classes[i]}', iou, sync_dist=True)
        self.log('mean/precison', mp, sync_dist=True)
        self.log('mean/recall', mr, sync_dist=True)
        self.log('mean/f1_score', mf1, sync_dist=True)
        self.log('mean/iou', miou, sync_dist=True)
    

if __name__ == '__main__':
    
    data_root = '/root/remote_sensing/second_dataset'
    mode = ['train', 'val']
    n_splits = 0
    devices = [0, 1, 2]
    max_epochs = 200
    seed = 3407
    batch_size = 12
    strategy = 'ddp_find_unused_parameters_true'
    
    data_model = KFlodDataModule(data_root, mode=mode, batch_size=batch_size, n_splits=n_splits)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    trainer = pl.Trainer(
        devices=devices,
        max_epochs=max_epochs,
        strategy=strategy
    )
    model = LitSeg()
    
    if n_splits == 0:
        trainer.fit(model, datamodule=data_model)
    else:
        for fold, (train_indices, val_indices) in enumerate(data_model.kfold.split(data_model.dataset)):
            print(f"KFold {fold + 1}")
            data_model.train_fold(train_indices)
            data_model.val_fold(val_indices)
            trainer.fit(model=model, datamodule=data_model)

