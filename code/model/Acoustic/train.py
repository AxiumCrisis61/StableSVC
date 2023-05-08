from utils import SVCDataset, EMA, get_standardizer, save_checkpoint, load_checkpoint
from ddpm import GaussianDiffusionTrainer, GaussianDiffusionSampler
from conversion_model import DiffusionConverter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
from torch.optim import AdamW
from argparse import ArgumentParser
import warnings
import os
import sys
sys.path.append("../../")
from config import CHECKPOINT_PATH_ACOUSTIC


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ################################### settings and hyper-parameters ###################################
    arg_parser = ArgumentParser(description='arguments for training DDPM')

    # framework settings
    arg_parser.add_argument('--framework', type=str, choices=('simple_diffusion',), default='simple_diffusion',
                            help='choice of conversion framework')
    arg_parser.add_argument('--use-ema', type=bool, default=True,
                            help='whether to use Exponential Moving Average to the model')

    # training settings
    arg_parser.add_argument('--resume', type=bool, default=True,
                            help='whether to resume training from the latest checkpoint')
    arg_parser.add_argument('--val-interval', type=int, default=200,
                            help='validation interval (steps); set as 0 to cancel validation')
    arg_parser.add_argument('--checkpoint-interval', type=int, default=100,
                            help='checkpoint interval (steps); set as 0 to cancel checkpointing')
    arg_parser.add_argument('--num-workers', type=int, default=4,
                            help='number of DataLoader workers')

    # training and validations set, and checkpoint
    arg_parser.add_argument('--training-set', type=str, choices=('Opencpop', 'M4Singer'), default='Opencpop',
                            help='the training set of the model')
    arg_parser.add_argument('--validation-set', type=str, choices=('Opencpop', 'M4Singer'), default='Opencpop',
                            help='the validation set of the model')
    arg_parser.add_argument('--ckpt-dir', type=str, default=CHECKPOINT_PATH_ACOUSTIC,
                            help='checkpoint path for the acoustic model')

    # training hyper-parameters
    arg_parser.add_argument('--epochs', type=int, default=100,
                            help='number of training epochs')
    arg_parser.add_argument('--batch-size', type=int, default=8,
                            help='batch size for mini-batch optimization')

    # AdamW optimizer hyper-parameters
    arg_parser.add_argument("--lr", type=float, default=5e-5,
                            help="initial learning rate")
    arg_parser.add_argument('--beta1', type=float, deault=0.9,
                            help='beta_1 for AdamW optimizer')
    arg_parser.add_argument('--beta2', type=float, deault=0.999,
                            help='beta_2 for ')
    arg_parser.add_argument('--weight-decay', type=float, deault=1e-2,
                            help='weight decay coefficient for AdamW optimizer')
    arg_parser.add_argument('--use-grad-clip', type=bool, deault=True,
                            help='gradient clip ceiling')
    arg_parser.add_argument('--grad-clip', type=float, deault=1,
                            help='gradient clip ceiling')

    args = arg_parser.parse_args()

    # ################################### training ###################################
    # create checkpoint path if none
    ckpt_path = os.path.join(args.ckpt_dir, args.framework)
    os.makedirs(ckpt_path, exist_ok=True)

    # get device (only single GPU is supported currently)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    train_set = SVCDataset(args.training_set, 'train')
    val_set = SVCDataset(args.validation_set, 'test')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=True)
    mel_standardizer, _, _ = get_standardizer()

    # models and optimizer
    model = DiffusionConverter().to(device)
    if args.use_ema:
        ema = EMA(model)
    else:
        ema = None
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # create diffusion framework
    if args.framework == 'simple_diffusion':
        ddpm_trainer = GaussianDiffusionTrainer(model)
        if args.use_ema:
            ddpm_sampler = GaussianDiffusionSampler(ema.shadow)
        else:
            ddpm_sampler = GaussianDiffusionSampler(model)
    else:
        raise ValueError("Unsupported conversion framework")

    # load checkpoints
    if args.resume:
        if os.path.isfile(os.path.join(ckpt_path, 'latest')):
            state_dict = load_checkpoint(ckpt_path, device)
            model.load_state_dict(state_dict['model'])
            if args.use_ema:
                ema.load_state_dict(state_dict['ema'])
            optimizer.load_state_dict(state_dict['optimizer'])
            epoch = state_dict['epoch'] + 1
            step = state_dict['step'] + 1
            best_val_error = state_dict['best_val_error']
            print('Resuming training from epoch {}, step{}...'.format(epoch, step))
            flag_continue = True
        else:
            flag_continue = False
            print('Starting training from scratch...')
            step = 0
            best_val_error = np.inf
    else:
        flag_continue = False
        print('Starting training from scratch...')
        step = 0
        best_val_error = np.inf

    # train
    ddpm_trainer.train()
    for epoch in range(args.epochs):
        print('-'*15 + f'epoch {epoch}' + '-'*15)
        for x, whisper, f0, loudness in train_loader:
            model.train()
            ema.train()

            step += 1
            optimizer.zero_grad()

            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            whisper = torch.autograd.Variable(whisper.to(device, non_blocking=True))
            f0 = torch.autograd.Variable(f0.to(device, non_blocking=True))
            loudness = torch.autograd.Variable(loudness.to(device, non_blocking=True))

            # calculate loss
            loss = ddpm_trainer(x, whisper=whisper, f0=f0, loudness=loudness)
            loss.backward()
            # gradient norm clipping
            if args.use_grad_clip:
                clip_grad_norm(model.parameters(), args.grad_clip)
            # optimization
            optimizer.step()
            # update EMA model
            ema.update()

            # dumping cuda memory
            del x, whisper, f0, loudness
            for i in range(5):
                torch.cuda.empty_cache()

            # save latest checkpoint
            if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
                save_checkpoint(os.path.join(ckpt_path, 'latest'), {
                    'model': model.state_dict(),
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'best_val_error': best_val_error
                })

            # validation and save best checkpoint
            if args.val_interval > 0 and step % args.val_interval == 0:
                model.eval()
                ema.eval()
                val_error_list = []

                for y_val, whisper_val, f0_val, loudness_val in val_loader:
                    y_val.to(device)
                    whisper_val.to(device)
                    f0_val.to(device)
                    loudness_val.to(device)

                    noise = torch.randn_like(y_val)

                    x_val = ddpm_sampler(noise, whisper=whisper_val, f0=f0_val, loudness=loudness_val)

                    val_error_list.append(F.mse_loss(x_val, y_val))

                val_error = np.mean(val_error_list)

                if val_error < best_val_error:
                    save_checkpoint(os.path.join(ckpt_path, 'best'), {
                        'model': model.state_dict(),
                        'ema': ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step,
                    })
