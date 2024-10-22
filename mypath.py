class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'data_voc':
            return '/root/桌面/pytorch-deeplab-xception-master/shiguanai'
        elif dataset == 'shiguanai':
            return '/root/data1/newcode/Voc'
        elif dataset == 'kvasir':
            return '/root/桌面/newcode11/kvasir-seg'
        
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
