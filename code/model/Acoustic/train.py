from utils import SVCDataset, EMA, save_checkpoint, load_checkpoint
from ddpm import GaussianDiffusionTrainer, GaussianDiffusionSampler
from conversion_model import DiffusionConverter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from argparse import ArgumentParser
import time
import warnings
import os
import sys
sys.path.append("../../")
from config import CKPT_ACOUSTIC, FRAMEWORK, USE_EMA, NOISE_SCHEDULE


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # ################################### settings and hyper-parameters ###################################
    arg_parser = ArgumentParser(description='arguments for training DDPM')

    # framework settings
    arg_parser.add_argument('--framework', type=str, choices=('simple_diffusion',), default=FRAMEWORK,
                            help='choice of conversion framework')
    arg_parser.add_argument('--use-ema', type=bool, default=USE_EMA,
                            help='whether to use Exponential Moving Average to the model')
    arg_parser.add_argument('--noise-schedule', type=str, choices=('linear', 'cosine'), default=NOISE_SCHEDULE,
                            help='noise scheduling strategy for the DDPM')

    # training settings
    arg_parser.add_argument('--resume', type=bool, default=True,
                            help='whether to resume training from the latest checkpoint')
    arg_parser.add_argument('--val-interval', type=int, default=200,
                            help='validation interval (steps); set as 0 to cancel validation')
    arg_parser.add_argument('--checkpoint-interval', type=int, default=20,
                            help='checkpoint interval (steps); set as 0 to cancel checkpointing')
    arg_parser.add_argument('--print-interval', type=int, default=5,
                            help='checkpoint interval (steps); set as 0 to cancel printing training information')

    # training and validations set, and checkpoint
    arg_parser.add_argument('--training-set', type=str, choices=('Opencpop', 'M4Singer'), default='Opencpop',
                            help='the training set of the model')
    arg_parser.add_argument('--validation-set', type=str, choices=('Opencpop', 'M4Singer'), default='Opencpop',
                            help='the validation set of the model')
    arg_parser.add_argument('--ckpt-dir', type=str, default=CKPT_ACOUSTIC,
                            help='checkpoint path for the acoustic model')

    # training hyper-parameters
    arg_parser.add_argument('--epochs', type=int, default=100,
                            help='number of training epochs')
    arg_parser.add_argument('--batch-size', type=int, default=8,
                            help='batch size for mini-batch optimization')
    arg_parser.add_argument('--val-batch-size', type=int, default=4,
                            help='batch size for validation')
    arg_parser.add_argument('--val-batch-num', type=int, default=4,
                            help='number of batches for one time of validation. set as 0 to use full val set,'
                                 '[NOTE] the validation process of DDPM is slow')

    # AdamW optimizer hyper-parameters
    arg_parser.add_argument("--lr", type=float, default=5e-5,
                            help="initial learning rate")
    arg_parser.add_argument('--beta1', type=float, default=0.9,
                            help='beta_1 for AdamW optimizer')
    arg_parser.add_argument('--beta2', type=float, default=0.999,
                            help='beta_2 for ')
    arg_parser.add_argument('--weight-decay', type=float, default=1e-2,
                            help='weight decay coefficient for AdamW optimizer')
    arg_parser.add_argument('--use-grad-clip', type=bool, default=True,
                            help='gradient clip ceiling')
    arg_parser.add_argument('--grad-clip', type=float, default=1,
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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, drop_last=True)

    # models and optimizer
    model = DiffusionConverter().to(device)
    if args.use_ema:
        ema = EMA(model).to(device)
    else:
        ema = None
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    # create diffusion framework
    if args.framework == 'simple_diffusion':
        ddpm_trainer = GaussianDiffusionTrainer(model, noise_schedule=args.noise_schedule).to(device)
        if args.use_ema:
            ddpm_sampler = GaussianDiffusionSampler(ema.shadow, noise_schedule=args.noise_schedule).to(device)
        else:
            ddpm_sampler = GaussianDiffusionSampler(model, noise_schedule=args.noise_schedule).to(device)
    else:
        raise ValueError("Unsupported conversion framework")

    # load checkpoints
    if args.resume:
        if os.path.isfile(os.path.join(ckpt_path, 'latest')):
            state_dict = load_checkpoint(os.path.join(ckpt_path, 'latest'), device)
            model.load_state_dict(state_dict['model'])
            if args.use_ema:
                ema.load_state_dict(state_dict['ema'])
            optimizer.load_state_dict(state_dict['optimizer'])
            last_epoch = state_dict['epoch']
            step = state_dict['step'] + 1
            best_val_error = state_dict['best_val_error']
            print('Resuming training from epoch {}, step{}...'.format(last_epoch, step))
            flag_continue = True
        else:
            flag_continue = False
            print('Starting training from scratch...')
            step = 0
            last_epoch = 1
            best_val_error = np.inf
    else:
        flag_continue = False
        print('Starting training from scratch...')
        step = 0
        last_epoch = 1
        best_val_error = np.inf

    # train
    ddpm_trainer.train()
    for epoch in range(last_epoch, args.epochs + 1):
        start = time.time()
        print('-'*15 + f'epoch {epoch}' + '-'*15)
        for x, whisper, f0, loudness in train_loader:
            model.train()
            if args.use_ema:
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
                clip_grad_norm_(model.parameters(), args.grad_clip)
            # optimization
            optimizer.step()
            # update EMA model
            if args.use_ema:
                ema.update()

            # print training information
            if args.print_interval > 0 and step % args.print_interval == 0:
                print('Step: {}, loss: {}'.format(step, loss.item()))

            # dumping cuda memory
            del x, whisper, f0, loudness
            for i in range(5):
                torch.cuda.empty_cache()

            # save latest checkpoint
            if args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0:
                if args.use_ema:
                    save_checkpoint(os.path.join(ckpt_path, 'latest'), {
                        'model': model.state_dict(),
                        'ema': ema.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'best_val_error': best_val_error
                    })
                else:
                    save_checkpoint(os.path.join(ckpt_path, 'latest'), {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'best_val_error': best_val_error
                    })

            # validation and save best checkpoint
            if args.val_interval > 0 and step % args.val_interval == 0:
                model.eval()
                if args.use_ema:
                    ema.eval()
                val_error_list = []
                iter_val = 0

                for y_val, whisper_val, f0_val, loudness_val in val_loader:
                    iter_val += 1
                    if iter_val > args.val_batch_num:
                        break
                    y_val = y_val.to(device)
                    whisper_val = whisper_val.to(device)
                    f0_val = f0_val.to(device)
                    loudness_val = loudness_val.to(device)

                    noise = torch.randn_like(y_val).to(device)

                    x_val = ddpm_sampler(noise, whisper=whisper_val, f0=f0_val, loudness=loudness_val)

                    with torch.no_grad():
                        val_error_list.append(F.mse_loss(x_val, y_val).cpu().numpy())

                    del y_val, whisper_val, f0_val, loudness_val, noise, x_val
                    for i in range(5):
                        torch.cuda.empty_cache()

                val_error = np.mean(val_error_list)
                print('Validation error at step {}: {}'.format(step, val_error))

                if val_error < best_val_error:
                    best_val_error = val_error
                    if args.use_ema:
                        save_checkpoint(os.path.join(ckpt_path, 'best'), {
                            'model': model.state_dict(),
                            'ema': ema.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'step': step,
                            'best_val_error': best_val_error,
                        })
                    else:
                        save_checkpoint(os.path.join(ckpt_path, 'best'), {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'step': step,
                            'best_val_error': best_val_error,
                        })

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
