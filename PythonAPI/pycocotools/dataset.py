from .cocoz import AnnZ, ImageZ, COCOZ   # 载入 cocoz


class Loader:
    def __init__(self, dataType, catNms, root, annType):
        self.annZ = AnnZ(root, annType)
        self.imgZ = ImageZ(root, f'images/{dataType}')
        self.dataType = dataType
        self.coco = self._coco()
        self.catNms = catNms

    def _coco(self):
        annFile = f'annotations/instances_{self.dataType}.json'
        return COCOZ(self.annZ, annFile)

    def det_labels(self, index):
        '''
        instances 的标签：(x_1,y_1,w,h,category_id)
        参考：COCO数据集的标注格式：https://zhuanlan.zhihu.com/p/29393415
        '''
        labels = []
        for ann in self.coco.imgToAnns[index]:
            labels.append(ann['bbox'] + [ann['category_id']])
        return labels

    @property
    def images(self):
        catIds = self.coco.getCatIds(self.catNms)  # 获取 Cat 的 Ids
        imgIds = self.coco.getImgIds(catIds=catIds)  #
        return self.coco.loadImgs(imgIds)

    def image2dataset(self, img):
        fname = img['file_name']
        fname = (f'{self.dataType}/{fname}')
        return self.imgZ.buffer2array(fname), self.det_labels(img['id'])

    def __getitem__(self, item):
        imgs = self.images[item]
        if isinstance(item, slice):
            return [self.image2dataset(img) for img in imgs]
        else:
            return self.image2dataset(imgs)

    def __len__(self):
        return len(self.images)