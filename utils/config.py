# utils/config.py
import argparse
import yaml
import sys
import os

def parse_args(description='NAT-HGT Project'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', dest='cfg_file', help='Config file path', default='configs/full_experiment.yaml', type=str)
    parser.add_argument('opts', help='Command line override', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()

def _merge_config(config, opts):
    if not opts: return config
    if len(opts) % 2 != 0: raise ValueError(f"opts must be key-value pairs, got: {opts}")
    for k, v in zip(opts[0::2], opts[1::2]):
        keys = k.split('.')
        sub = config
        try:
            for key in keys[:-1]: sub = sub[key]
            orig = sub[keys[-1]]
            if isinstance(orig, bool): v = str(v).lower() in ('true', '1', 'yes')
            elif orig is None:
                try: v = float(v) if '.' in v else int(v)
                except: pass
            else: v = type(orig)(v)
            print(f"⚡️ Override: {k} = {v}")
            sub[keys[-1]] = v
        except Exception as e: print(f"❌ Error merging {k}: {e}")
    return config

def get_config(description='NAT-HGT Project'):
    args = parse_args(description)
    if not os.path.exists(args.cfg_file):
        print(f"❌ Config not found: {args.cfg_file}")
        sys.exit(1)
    with open(args.cfg_file, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return _merge_config(cfg, args.opts)