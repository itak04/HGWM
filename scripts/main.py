import argparse
import yaml
from pathlib import Path
from ablation_protocol import ABLATIONS, apply_ablation, write_run_manifest

from dotenv import load_dotenv
from api import *
from base_agent import *
from hgwm_agent import *
from hgwm_env import *
from custom_agent import *
from custom_env import *

load_dotenv()
def set_seed(seed: int):
    # 设置 random 模块的种子
    random.seed(seed)
    # 设置 numpy 模块的种子
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Run a dynamic envmark")

    parser.add_argument('--config', type=str, default='HGWM', help='name of the YAML config file')
    parser.add_argument('-n', '--name', type=str, help='Name for the run (optional)')
    parser.add_argument('-lf', '--log_freq', type=int, help='Logging frequency (optional)')
    parser.add_argument('-ms', '--max_steps', type=int, help='Max steps per episode (optional)')
    parser.add_argument('-ne', '--num_episodes', type=int, help='Number of episodes to run (optional)')
    parser.add_argument('--episodes', type=str, help='Comma-separated list of specific episode indices to test (e.g., "994,964")')
    parser.add_argument('-pa', '--parallel', action='store_true', help='Enable parallel execution')
    parser.add_argument('--instances', type=int, help='Number of instances for parallel execution (optional)')
    parser.add_argument('--instance', type=int, help='Instance number for parallel execution (optional)')
    parser.add_argument('--port', type=int, help='port number for Flask server parallel execution (optional)')
    parser.add_argument('--dataset', type=str, default='hm3d_v0.2', choices=['hm3d_v0.1', 'hm3d_v0.2', 'mp3d'], help='Type of dataset')
    parser.add_argument(
        '--ablation',
        default='full',
        choices=sorted(ABLATIONS),
        help='Reviewer-requested leave-one-component-out or weight-sensitivity variant',
    )

    parser.add_argument('--seed', type=int, default=48, help='Python/NumPy seed; remote inference may remain stochastic')
    args = parser.parse_args()
    set_seed(args.seed)

    # Load configuration from YAML file
    with open(f'config/{args.config}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config = apply_ablation(config, args.ablation)
    config.setdefault('env_cfg', {})['seed'] = args.seed
    config['env_cfg']['ablation'] = args.ablation
    # Override YAML config with command-line arguments if provided
    if args.name is not None:
        config['env_cfg']['name'] = args.name
    elif args.ablation != 'full':
        config['env_cfg']['name'] = f"{config['env_cfg'].get('name', 'default')}_{args.ablation}"
    if args.log_freq is not None:
        config['env_cfg']['log_freq'] = args.log_freq
    if args.max_steps is not None:
        config['env_cfg']['max_steps'] = args.max_steps
    if args.num_episodes is not None:
        config['env_cfg']['num_episodes'] = args.num_episodes
    if args.episodes is not None:
        # Parse specific episodes list
        episodes_list = [int(x.strip()) for x in args.episodes.split(',')]
        config['env_cfg']['specific_episodes'] = episodes_list
        config['env_cfg']['num_episodes'] = len(episodes_list)  # Override number of episodes
    if args.instances is not None:
        config['env_cfg']['instances'] = args.instances
    if args.instance is not None:
        config['env_cfg']['instance'] = args.instance
    if args.port is not None:
        config['env_cfg']['port'] = args.port
    if args.parallel:
        config['env_cfg']['parallel'] = True
    if args.dataset:
        config['env_cfg']['dataset'] = args.dataset
    
    env_cls = globals()[config['env_cls']]
    env = env_cls(cfg=config)
    manifest = write_run_manifest(Path(env.log_file).with_suffix('.manifest.json'), config, args.seed, args.ablation, Path(__file__).resolve().parents[1])
    import hashlib, json
    env.source_fingerprint = hashlib.sha256(json.dumps(manifest['source_sha256'], sort_keys=True).encode()).hexdigest()
    env.run_experiment()

if __name__ == '__main__':
    set_seed(48)
    main()
