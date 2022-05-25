from uread.transforms import CharBlock, CharTokenizer


def test_char_tokenizer():
    tokenizer = CharTokenizer()
    test_string = "test"
    tokens = tokenizer(test_string)
    assert tokens == ['<', 't', 'e', 's', 't', '>']


def test_char_block():
    block = CharBlock()
    assert isinstance(block.type_tfms[0], CharTokenizer)