import argparse
import warnings
import torch.utils.data
from utility import *
from hsi_setup import train_options, Engine
from functools import partial
from torchvision.transforms import Compose

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    """Train Settings"""
    parser = argparse.ArgumentParser(description="Hyperspectral Image Denosing")
    opt = train_options(parser)

    print(f'opt settings: {opt}')

    """Set Random Status"""
    seed_everywhere(opt.seed)


    """Setup Engine"""
    engine = Engine(opt)

    print('model params: %.2f' % (sum([t.nelement() for t in engine.net.parameters()])/10**6))
    print('==> Preparing data..')
    """Training Data Settings"""

    HSI2Tensor = partial(HSI2Tensor, use_2dconv = engine.get_net().use_2dconv)

    common_transform = lambda x: x

    common_transform_1 = lambda x: x
    common_transform_2 = Compose([
        lambda x: x,
        partial(rand_crop, cropx=32, cropy=32),
    ])

    target_transform = Compose([
        lambda x: x,
        HSI2Tensor(),
    ])


    train_transform = Compose([
        AddNoiseBlindv1(10, 70),
        HSI2Tensor()
    ])

    repeat = 5
    trainsets = make_dataset(
        opt, train_transform,
        target_transform, common_transform, 8, repeat)


    """Validation Data Settings"""
    basefolder = './data/CAVE/val'
    mat_names = ['gauss_30', 'gauss_50']

    mat_datasets = [MatDataFromFolder(os.path.join(
        basefolder, name)) for name in mat_names]

    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[:, ...][None]),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                       transform=lambda x:x),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

    mat_loaders = [torch.utils.data.DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=opt.no_cuda
    ) for mat_dataset in mat_datasets]

    """Main loop"""
    base_lr = opt.lr
    if not opt.resume:
        adjust_learning_rate(engine.optimizer, base_lr)

    epoch_per_save = 10
    rand_status_list = np.random.get_state()[1].tolist()

    while engine.epoch < 50:
        np.random.seed(rand_status_list[engine.epoch % len(rand_status_list)])  # reset seed per epoch, otherwise the noise will be added with a specific pattern

        engine.train(trainsets)

        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()

        for mat_name, mat_loader in zip(mat_names, mat_loaders):
            avg_psnr, avg_loss = engine.validate(mat_loader, mat_name)

        if avg_psnr > engine.best_psnr:
            engine.best_psnr = avg_psnr
            model_best_path = os.path.join(engine.basedir, engine.prefix, 'model_best.pth')
            print('')
            engine.save_checkpoint(
                model_out_path=model_best_path
            )
