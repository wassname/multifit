from typing import Iterator, Collection
from fastai.data_block import CategoryListBase
from fastai.text import *


class BinaryProcessor(CategoryProcessor):
    def create_classes(self, classes):
        self.classes = classes
        if classes is not None: self.c2i = {0:0, 1:1}
    def generate_classes(self, items):
        return [0]

class BinaryCategoryList(CategoryListBase):
    "Basic `ItemList` for single classification labels."
    _processor=BinaryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, label_delim:str=None, **kwargs):
        super().__init__(items, classes=classes, **kwargs)
        mean = self.items.mean()
        if mean and mean!=0:
            weight = torch.tensor([1 / self.items.mean()]).cuda()
        else:
            weight = None
            raise Exception('debug')
        self.loss_func = BCEWithLogitsFlat(weight=weight)

    def reconstruct(self, t):
        return Category(t, self.c2i[t.item()])

    def get(self, i):
        o = self.items[i]
        if o is None: return None
        return Category(o, self.c2i[o])

    def analyze_pred(self, pred, thresh:float=0.5): return pred.argmax()
