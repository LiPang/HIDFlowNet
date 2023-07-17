import argparse
import warnings

import torch.utils.data
from utility import *
from hsi_setup import train_options, Engine
from functools import partial
from torchvision.transforms import Compose
sigmas = [10, 30, 50, 70]
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
    # print(engine.net)
    print('model params: %.2f' % (sum([t.nelement() for t in engine.net.parameters()])/10**6))

    print('==> Preparing data..')

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor, use_2dconv=engine.get_net().use_2dconv)

    add_noniid_noise = Compose([
        AddNoiseNoniid(sigmas),
        SequentialSelect(
            transforms=[
                lambda x: x,
                AddNoiseImpulse(),
                AddNoiseStripe(),
                AddNoiseDeadline()
            ]
        )
    ])


    common_transform = lambda x: x

    target_transform = HSI2Tensor()

    train_transform = Compose([
        add_noniid_noise,
        HSI2Tensor()
    ])

    print('==> Preparing data..')

    repeat = 5
    icvl_64_31_TL = make_dataset(
        opt, train_transform,
        target_transform, common_transform, 8, repeat)

    """Test-Dev"""
    basefolder = './data/CAVE/val'
    mat_names = ['complex_noniid', 'complex_mixture']

    mat_datasets = [MatDataFromFolder(os.path.join(
        basefolder, name)) for name in mat_names]

    if not engine.get_net().use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                       transform=lambda x: x[:, ...][None]),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt'),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

    mat_loaders = [torch.utils.data.DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt.no_cuda
    ) for mat_dataset in mat_datasets]

    base_lr = opt.lr
    epoch_per_save = 10
    adjust_learning_rate(engine.optimizer, opt.lr)

    rand_status_list = np.random.get_state()[1].tolist()
    # from epoch 50 to 100
    if opt.clear:
        engine.best_psnr = 0
        engine.epoch = 50
    while engine.epoch < 100:
        np.random.seed(rand_status_list[engine.epoch % len(rand_status_list)])

        engine.train(icvl_64_31_TL)

        display_learning_rate(engine.optimizer)
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path
        )

        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()

        for mat_name, mat_loader in zip(mat_names, mat_loaders):
            avg_psnr, avg_loss = engine.validate(mat_loader, mat_name)

        if avg_psnr > engine.best_psnr:
            engine.best_psnr = avg_psnr
            model_best_path = os.path.join(engine.basedir, engine.prefix, 'model_best.pth')
            engine.save_checkpoint(
                model_out_path=model_best_path
            )