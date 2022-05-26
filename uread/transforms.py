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


class CharBlock(TransformBlock):
    "A `TransformBlock` for individual characters"
    @delegates(Numericalize.__init__)
    def __init__(self, vocab=None, seq_len=72, **kwargs):
        import pdb ; pdb.set_trace()
        type_tfms = [CharTokenizer(), Numericalize(vocab, **kwargs)]
        return super().__init__(
            type_tfms=type_tfms,
            dls_kwargs={'before_batch': Pad_Chunk(seq_len=seq_len)}
        )

