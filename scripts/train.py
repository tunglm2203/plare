import argparse
import os

from research.utils.config import Config, str2bool


def try_wandb_setup(path, config):
    try:
        import wandb
    except ImportError:
        return
    project_dir = os.path.dirname(os.path.dirname(__file__))
    exp_dir = os.path.join(project_dir, path)
    project = "YOUR PROJECT"
    entity = "tungluu2203"
    wandb.login(key="local-8e01dce497b881311ad295b0ac3a69940a16de30")

    if config.config['env'] is not None:
        tags = [config.config['env']]
    elif config.config['eval_env'] is not None:
        tags = [config.config['eval_env']]
    else:
        raise ValueError
    wandb_offline = 'disabled' if 'debug' in path else 'online'
    group = os.path.basename(os.path.dirname(exp_dir))
    name = os.path.basename(exp_dir)
    wandb.init(
        name=name,
        group=group,
        tags=tags,
        dir=exp_dir,
        project=project,
        entity=entity,
        config=config.flatten(separator="-"),
        mode=wandb_offline
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--path", "-p", type=str, default=None)
    parser.add_argument("--device", "-d", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)

    parser.add_argument("--logdir", type=str, default="training_logs")
    parser.add_argument("--exp_name", type=str, default="plare")
    parser.add_argument("--ignore_unsure", type=str2bool, default=False)
    parser.add_argument("--dropout", type=float, default=0.25)
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.seed is not None:
        config.config['seed'] = args.seed

    if args.env_name is not None:
        if 'env' in config.config and config.config['env'] is not None:
            # This requires .yaml config for online training should declare an environment
            config.config['env'] = args.env_name
        if 'eval_env' in config.config:
            config.config['eval_env'] = args.env_name
    if args.data_path is not None:
        config.config['dataset_kwargs']['path'] = args.data_path

    if 'ignore_unsure' in config.config['dataset_kwargs']:
        config.config['dataset_kwargs']['ignore_unsure'] = args.ignore_unsure
    if 'dropout' in config.config['network_kwargs']['actor_kwargs']:
        config.config['network_kwargs']['actor_kwargs']['dropout'] = args.dropout
    print(config)

    sub_exp_name = args.exp_name + f"_s{config.config['seed']}"
    args.path = os.path.join(args.logdir, args.env_name, args.exp_name, sub_exp_name)
    os.makedirs(args.path, exist_ok=True)  # Change this to false temporarily so we don't recreate experiments
    try_wandb_setup(args.path, config)
    config.save(args.path)  # Save the config

    # Parse the config file to resolve names.
    config = config.parse()

    # Get everything at once.
    trainer = config.get_trainer(device=args.device)

    # Train the model
    trainer.train(args.path)
