import yaml
import sys

CONFIG_PATH = "config.yaml"

try:
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
except yaml.YAMLError as e:
    print(f"❌ YAML parsing error: {e}")
    sys.exit(1)
