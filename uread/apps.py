from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
from rich.console import Console
from functools import partial

from fastai.data.block import DataBlock
from fastai.data.transforms import ColReader, ColSplitter, RandomSplitter, Normalize
from fastai.metrics import accuracy_multi, accuracy
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize, ResizeMethod
from fastai.callback.hook import num_features_model
from fastai.vision.learner import create_body, create_cnn_model
from fastai.vision.core import imagenet_stats

import pandas as pd
import fastapp as fa
from fastapp.vision import VisionApp
from fastapp.examples.image_classifier import PathColReader
from torchvision import models

from .transforms import CharBlock
from .modules import CharDecoder


console = Console()


class Uread(fa.FastApp):
    def __init__(self):
        super().__init__()
        self.vocab = list("_<abcdefghijklmnopqrstuvwxyz>") # The first char is the padding

    def dataloaders(
        self,
        csv: Path = fa.Param(default=None, help="A CSV with image paths and the text to read."),
        image_column: str = fa.Param(default="image", help="The name of the column with the image paths."),
        text_column: str = fa.Param(
            default="text", help="The name of the column with the text of the image."
        ),
        base_dir: Path = fa.Param(default=None, help="The base directory for images with relative paths. If empty, then paths are relative to the CSV file."),
        validation_column: str = fa.Param(
            default="validation", 
            help="The column in the dataset to use for validation. "
                "If the column is not in the dataset, then a validation set will be chosen randomly according to `validation_proportion`.",
        ),
        validation_proportion: float = fa.Param(
            default=0.2, 
            help="The proportion of the dataset to keep for validation. Used if `validation_column` is not in the dataset."
        ),
        batch_size: int = fa.Param(default=16, help="The number of items to use in each batch."),
        width: int = fa.Param(default=224, help="The width to resize all the images to."),
        height: int = fa.Param(default=224, help="The height to resize all the images to."),
        resize_method:str = fa.Param(default="squish", help="The method to resize images."),
        max_images:int = fa.Param(default=None, help="If set, then this is the maximum number of images used for training/validation."),
    ):
        df = pd.read_csv(csv)
        if max_images:
            df = df.sample(max_images, random_state=max_images)

        if not base_dir:
            base_dir = Path(csv).parent

        # Create splitter for training/validation images
        if validation_column and validation_column in df:
            splitter = ColSplitter(validation_column)
        else:
            splitter = RandomSplitter(validation_proportion)

        self.datablock = DataBlock(
            blocks=[ImageBlock, CharBlock(vocab=self.vocab)],
            get_x=PathColReader(column_name=image_column, base_dir=base_dir),
            get_y=ColReader(text_column),
            splitter=splitter,
            item_tfms=Resize( (height, width), method=resize_method ),
            batch_tfms=Normalize.from_stats(*imagenet_stats),
        )

        dataloaders = self.datablock.dataloaders(df, bs=batch_size)
        
        batch = dataloaders.one_batch()
        print("Batch shapes:", batch[0].shape, batch[1].shape)

        return dataloaders

    def model(
        self,
    ) -> nn.Module:
        """
        Creates a deep learning model for the app to use.

        Returns:
            nn.Module: The created model.
        """
        size = 512
        encoder = create_cnn_model(models.resnet18, size)
        self.fine_tune = True
        # features = num_features_model(encoder)
        decoder = CharDecoder(input_size=size, vocab_size=len(self.vocab))

        return nn.Sequential(
            encoder,
            decoder,
        )

    def metrics(self):
        return [partial(accuracy, axis=-2)]

    def monitor(self):
        return "accuracy"

    def loss_func(self):
        return nn.CrossEntropyLoss()
