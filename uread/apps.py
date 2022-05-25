from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
from rich.console import Console

from fastai.data.block import DataBlock
from fastai.data.transforms import ColReader, ColSplitter, RandomSplitter
from fastai.metrics import accuracy
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize, ResizeMethod

import pandas as pd
import fastapp as fa
from fastapp.vision import VisionApp
from fastapp.examples.image_classifier import PathColReader

from .transforms import CharBlock

console = Console()


class Uread(fa.FastApp):

    def dataloaders(
        self,
        csv: Path = fa.Param(default=None, help="A CSV with image paths and the text to read."),
        image_column: str = fa.Param(default="image", help="The name of the column with the image paths."),
        text_column: str = fa.Param(
            default="text", help="The name of the column with the text of the image."
        ),
        base_dir: Path = fa.Param(default="./", help="The base directory for images with relative paths."),
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
    ):
        df = pd.read_csv(csv)

        # Create splitter for training/validation images
        if validation_column and validation_column in df:
            splitter = ColSplitter(validation_column)
        else:
            splitter = RandomSplitter(validation_proportion)

        datablock = DataBlock(
            blocks=[ImageBlock, CharBlock],
            get_x=PathColReader(column_name=image_column, base_dir=base_dir),
            get_y=ColReader(text_column),
            splitter=splitter,
            item_tfms=Resize( (height, width), method=resize_method ),
        )

        return datablock.dataloaders(df, bs=batch_size)

    def model(
        self,
    ) -> nn.Module:
        """
        Creates a deep learning model for the Pomona to use.

        Returns:
            nn.Module: The created model.
        """
        raise NotImplemented("Model function not implemented yet.") 
        return nn.Sequential(
        )

    def metrics(self):
        return [accuracy]

    def monitor(self):
        return "accuracy"
