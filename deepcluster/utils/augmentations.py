import argparse

from torchvision import transforms


def get_augmentation_fn(args: argparse.Namespace):

    augmentations = []

    if args.augmentation_resize is not None:
        augmentations.append(transforms.Resize(args.augmentation_resize))

    if args.augmentation_random_crop is not None:
        augmentations.append(transforms.RandomCrop(args.augmentation_random_crop))

    if args.augmentation_random_rotation is not None:
        augmentations.append(
            transforms.RandomRotation(args.augmentation_random_rotation)
        )

    if args.augmentation_random_horizontal_flip:
        augmentations.append(transforms.RandomHorizontalFlip())

    if args.augmentation_random_vertical_flip:
        augmentations.append(transforms.RandomVerticalFlip())

    if args.augmentation_color_jitter:
        augmentations.append(
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        )

    if args.augmentation_random_autocontrast:
        augmentations.append(transforms.RandomAutocontrast())

    if args.augmentation_random_equalize:
        augmentations.append(transforms.RandomEqualize())

    augmentation_fn = transforms.Compose(
        [
            *augmentations,
            transforms.ToTensor(),
        ]
    )

    return augmentation_fn
