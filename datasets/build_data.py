from .cityscapes import Cityscapes
from .voc import VOCSegmentation
from datasets import extra_transforms as et


def build_dataset(args):
    if args.dataset.lower() == 'voc':

        train_transform = et.ExtCompose([
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(args.image_size, args.image_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(args.image_size),
            et.ExtCenterCrop(args.image_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = VOCSegmentation(root=args.data_root, image_set='train',
                                    transform=train_transform)
        val_dst = VOCSegmentation(root=args.data_root, image_set='val',
                                  transform=val_transform)

    elif args.dataset.lower() == 'cityscapes':

        train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(args.image_size, args.image_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            et.ExtResize(args.image_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=args.data_root, split='train',
                               transform=train_transform)
        val_dst = Cityscapes(root=args.data_root, split='val',
                             transform=val_transform)

    else:
        print('No support datasets!')
    return train_dst, val_dst