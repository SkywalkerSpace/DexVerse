conda activate dexverse

pip install -e packages/openpi-client

python scripts/zero_agent.py --task=Dexverse-OpenLaptop-v0 --enable_cameras --num_envs=1

python scripts/pi05_agent.py --task=Dexverse-OpenLaptop-v0 --num_envs=1

