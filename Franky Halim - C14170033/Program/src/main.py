import logging
from pytorch_lightning import Trainer
from extractive import ExtractiveSummarizer
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything


logger = logging.getLogger(__name__)

def main(args):
    if args.seed:
        seed_everything(args.seed, True)  
        args.deterministic = True
    
    model = ExtractiveSummarizer(hparams=args)
    lr_logger = LearningRateMonitor() 
    
    type_used_data = args.data_path.split('/')[-2].split('_')[-1]    
    if args.no_use_token_type_ids:
        temp_token_type_ids = "no-token-type-ids"
    else:
        temp_token_type_ids = "use-token-type-ids"
    temp_pooling_mode = args.pooling_mode.replace('_','-')
    
    if args.use_logger == "wandb":
        wandb_name = f"{args.model_name_or_path}_{type_used_data}_{temp_token_type_ids}_{temp_pooling_mode}_{args.classifier}_{args.classifier_transformer_num_layers}_{args.seed}_{args.learning_rate}_{args.classifier_dropout}_{args.batch_size}_{args.max_epochs}"
        wandb_logger = WandbLogger(
            name=wandb_name,project=args.wandb_project, log_model=(not args.no_wandb_logger_log_model)
        )
        args.logger = wandb_logger
    
    args.callbacks = [lr_logger]
    trainer = Trainer.from_argparse_args(args)

    if args.do_train:
        trainer.fit(model)
    if args.do_test:
        trainer.test(model)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--default_root_dir", type=str, default=None, help="Default path for logs and weights.",
    )
    parser.add_argument(
        "--weights_save_path",
        type=str,
        default=None,
        help="""Where to save weights if specified. Will override `--default_root_dir` for 
        checkpoints only.""",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=4,
        type=int,
        help="Limits training to a max number number of epochs",
    )
    parser.add_argument(
        "--min_steps",
        default=None,
        type=int,
        help="Limits training to a minimum number number of steps",
    )
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Limits training to a max number number of steps",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="""Accumulates grads every k batches.""",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        default=1,
        type=int,
        help="Check val every n train epochs.",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs to train on or Which GPUs to train on. (-1 = all gpus, 1 = only using one GPU)",
    )
    parser.add_argument(
        "--gradient_clip_val", default=1.0, type=float, help="Gradient clipping value (default gradient clipping algorithm is set to 'norm' and clip global norm to <=1.0)"   #https://github.com/PyTorchLightning/pytorch-lightning/issues/5671
    )  
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Runs 1 batch of train, test and val to find any bugs (ie: a sort of unit test).",
    )
    parser.add_argument(
        "--limit_train_batches",
        default=1.0,
        type=float,
        help="How much of training dataset to check. Useful when debugging or testing something that happens at the end of an epoch.",
    )
    parser.add_argument(
        "--limit_val_batches",
        default=1.0,
        type=float,
        help="How much of validation dataset to check. Useful when debugging or testing something that happens at the end of an epoch.",
    )
    parser.add_argument(
        "--limit_test_batches",
        default=1.0,
        type=float,
        help="How much of test dataset to check.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="Full precision (32). Can be used on CPU, GPU or TPUs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for reproducible results and fold id.",
    )
    parser.add_argument(
        "--profiler",
        default="simple",
        type=str,
        help="To profile individual steps during training and assist in identifying bottlenecks.",
    )
    parser.add_argument(
        "--progress_bar_refresh_rate",
        default=50,
        type=int,
        help="How often to refresh progress bar (in steps).",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        default=0,
        type=int,
        help="Sanity check runs n batches of val before starting the training routine. This catches any bugs in your validation without having to wait for the first validation check.",
    )
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        help="How often within one training epoch to check the validation set. Can specify as float or int. Use float to check within a training epoch. Use int to check every n steps (batches).",
    )
    parser.add_argument(
        "--use_logger",
        default="wandb",
        type=str,
        help="Which program to use for logging. Default to `wandb`.",
    )
    parser.add_argument(
        "--wandb_project",
        default="skripsi",
        type=str,
        help="The wandb project to save training runs.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Run the training procedure."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Run the testing procedure."
    )
    parser.add_argument(
        "--load_checkpoint",
        default=None,
        type=str,
        help="Loads the model weights and hyperparameters from a given checkpoint.",
    )
    parser.add_argument(
        "--no_wandb_logger_log_model",
        action="store_true",
        help="Only applies when using the `wandb` logger. Set this argument to NOT save checkpoints in wandb directory to upload to W&B servers.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.",
    )
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="weight decay for adam")
    parser.add_argument(
        "--optimizer_type",
        default="adamw",
        type=str,
        help="""Which optimizer to use: `adamw` (default)""",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.",
    )
    parser.add_argument(
        "--use_scheduler",
        default="linear",
        type=str,
        help="`linear`: Use a linear schedule that inceases linearly over `--warmup_steps` to `--learning_rate` then decreases linearly for the rest of the training process.",
    )
    parser.add_argument(
        "--log",
        dest="logLevel", # name of the attribute to be added to the object
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )

    parser = ExtractiveSummarizer.add_model_specific_args(parser)
    main_args = parser.parse_args()

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(main_args.logLevel),
    )
    
    # Train and Test
    main(main_args)
