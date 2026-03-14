import torch
import argparse
import numpy as np
from datetime import datetime
import datetime, platform, subprocess
import os
import wandb, json
from models.a3net import BaseModel
from modules.loss import compute_loss
from modules.trainer_cmn import Trainer
from modules.tokenizers import Tokenizer
from modules.metrics import compute_scores
from modules.dataloaders import R2DataLoader
from modules.optimizers import build_optimizer, build_lr_scheduler


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None

def get_env_info():
    return {
        "time": datetime.datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": get_git_commit(),
    }

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=1000, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args

# def main():
#     # parse arguments
#     args = parse_agrs()

#     os.makedirs(args.save_dir, exist_ok=True)
#     dump_json(os.path.join(args.save_dir, "run_config.json"), vars(args))
#     dump_json(os.path.join(args.save_dir, "env.json"), get_env_info())
#         # ---- W&B init (optional, but recommended) ----
#     use_wandb = True  # hoặc cho thành arg --use_wandb nếu muốn
#     if use_wandb:
#         wandb.init(
#             project=os.environ.get("WANDB_PROJECT", "A3Net-IU-Xray"),
#             name=os.environ.get("WANDB_NAME", f"{args.dataset_name}_date{time.now().strftime('%Y%m%d_%H%M%S')}"),
#             config=vars(args),
#         )
#         # log thêm thông tin môi trường bạn đã dump
#         wandb.config.update(get_env_info(), allow_val_change=True)

#     # fix random seeds
#     torch.manual_seed(args.seed)
#     torch.backends.dcunn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(args.seed)

#     # create tokenizer
#     tokenizer = Tokenizer(args)
    
#     # create data loader
#     train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
#     val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
#     test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

#     # build model architecture
#     model = BaseModel(args, tokenizer).to('cuda')

#     # get function handles of loss and metrics
#     criterion = compute_loss
#     metrics = compute_scores

#     # build optimizer, learning rate scheduler
#     optimizer = build_optimizer(args, model)
#     lr_scheduler = build_lr_scheduler(args, optimizer)
    
#     # build trainer and start to train
#     trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
#     trainer.train()

def main():

    args = parse_agrs()

    # ===== run id chuẩn research =====
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{args.dataset_name}_bs{args.batch_size}_lr{args.lr_ed}_seed{args.seed}_{run_time}"

    os.makedirs(args.save_dir, exist_ok=True)

    dump_json(os.path.join(args.save_dir, "run_config.json"), vars(args))
    dump_json(os.path.join(args.save_dir, "env.json"), get_env_info())

    # ===== W&B INIT =====
    wandb.init(
        project="A3Net-IU-Xray",
        name=run_name,
        group=f"beam{args.beam_size}_layer{args.num_layers}",
        tags=[
            args.dataset_name,
            args.visual_extractor,
            f"topk{args.topk}",
            f"cmm{args.cmm_size}"
        ],
        config=vars(args)
    )

    wandb.config.update(get_env_info(), allow_val_change=True)

    # ===== log git commit (rất mạnh khi research) =====
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        wandb.config.update({"git_commit": commit_id})
    except:
        pass

    # ===== fix seed chuẩn =====
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ===== tokenizer =====
    tokenizer = Tokenizer(args)

    # ===== dataloader =====
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # ===== log dataset info =====
    wandb.config.update({
        "train_size": len(train_dataloader.dataset),
        "val_size": len(val_dataloader.dataset),
        "test_size": len(test_dataloader.dataset),
        "vocab_size": len(tokenizer.idx2word)
    })

    # ===== model =====
    model = BaseModel(args, tokenizer).to('cuda')

    # ===== log model parameter count =====
    total_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"model_parameters": total_params})

    # ===== gradient tracking =====
    wandb.watch(model, log="gradients", log_freq=500)

    # ===== loss + metric =====
    criterion = compute_loss
    metrics = compute_scores

    # ===== optimizer + scheduler =====
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # ===== trainer =====
    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        args,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        test_dataloader
    )

    trainer.train()

    # ===== save artifact model =====
    artifact = wandb.Artifact("best_model", type="model")
    artifact.add_file(os.path.join(args.save_dir, "model_best.pth"))
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()