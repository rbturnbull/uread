import torch
from fastai.torch_core import TitledStr
from fastai.data.transforms import DisplayedTransform
from fastai.text.data import Pad_Chunk
from fastai.data.block import TransformBlock
from fastai.text.data import Numericalize
from fastcore.meta import delegates

class CharTokenizer(DisplayedTransform):
    input_types = (str,)
    def encodes(self, object:str): 
        return ["<"] + list(object.lower()) + [">"]

    def decodes(self, object): 
        return TitledStr("".join(object))


class PadZero(DisplayedTransform):
    "Pad `samples` by adding padding by chunks of size `seq_len`"
    def __init__(self, seq_len=72, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
    
    def encodes(self, x):
        difference = self.seq_len - x.shape[1]
        if difference < 0:
            return x[:self.seq_len]

        padded = torch.zeros(x.shape[0], self.seq_len)
        padded[:,difference:] = x

        return padded

    def decodes(self, object):
        object[object != 0]



class CharBlock(TransformBlock):
    "A `TransformBlock` for individual characters"
    @delegates(Numericalize.__init__)
    def __init__(self, vocab=None, seq_len=72, **kwargs):
        type_tfms = [CharTokenizer(), Numericalize(vocab, **kwargs)]
        return super().__init__(
            type_tfms=type_tfms,
            dls_kwargs={'before_batch': PadZero(seq_len=seq_len)}
        )


