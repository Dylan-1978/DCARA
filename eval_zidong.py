import argparse
import os
import numpy as np
from tqdm import tqdm
from utils.metrics import Evaluator
from dataloaders import make_data_loader
import torch
from modeling.DCARA.models import DCARA

def main():
    # ...（省略其他参数设置）...
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
    parser.add_argument('--weights-folder', type=str, default=r'/root/data1/2',
                        help='Directory containing model weights')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--dataset', type=str, default='shiguanai',
                        choices=['pascal', 'coco', 'cityscapes','data_voc','shiguanai'],)
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--batch-size', type=int, default=8,
                         help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--base-size', type=int, default=352,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=352,
                        help='crop image size')
    parser.add_argument('--model', type=str, default='DCARA',
                        choices=['DeepLab', 'FCBFormer','DCARA'],
                        help='model name (default: FCBformer)')
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet','FCBFormer',],
                        help='backbone name (default: resnet)')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader, val_loader,  nclass = make_data_loader(args, **kwargs)
    
    # ...（省略数据加载器和模型定义）...

    # Load the model
    if args.model == 'DCARA':
        model = DCARA(nclass=self.nclass,size=args.base_size)


    best_miou = -1
    best_miou_file = ''
    weight_files = [os.path.join(args.weights_folder, f) for f in os.listdir(args.weights_folder)
                    if f.endswith('.pth.tar')]
    print(weight_files)
    for weight_file in weight_files:
        # 加载模型权重
        ckpt = torch.load(weight_file, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        if args.cuda:
            model = model.cuda()
        model.eval()

        evaluator = Evaluator(nclass)
        evaluator.reset()

        tbar = tqdm(val_loader, desc='\r')
        for i, sample in enumerate(tbar):
                # ...（省略模型推断和评估代码）...
            image, target = sample['image'], sample['label']
            if args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU, mIou_per = evaluator.Mean_Intersection_over_Union()
        F1_score = evaluator.F1_score()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        Sp = evaluator.Specificity()
        Se = evaluator.Sensitivity()
        # ...（省略评价指标的计算和打印代码）...

        print('Results for model:', weight_file)
        print("Acc:{}\nAcc_class:{}\nmIoU:{}\nmIou_per:{}\nF1_score: {}\nFWIoU: {}\nSP: {}\nSE: {}".format(
            Acc, Acc_class, mIoU, mIou_per, F1_score, FWIoU, Sp, Se
        ))
# 检查当前mIoU是否是最好的，并在必要时更新记录
        if mIoU > best_miou:
            best_miou = mIoU
            best_miou_file = weight_file

    # 所有权重文件测试完成后，打印最高的mIoU结果
    print('Best mIoU result:')
    print("Best mIoU: {:.4f} from model: {}".format(best_miou, best_miou_file))

if __name__ == "__main__":
    main()
