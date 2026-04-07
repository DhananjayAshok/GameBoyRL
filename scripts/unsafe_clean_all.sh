echo "Uncomment to run"; exit 1
source scripts/utils.sh
bash scripts/clean_cache.sh
currpath=$(pwd)
cd $storage_dir
echo "Cleaning up all data folders in $storage_dir..."
rm -rf curiosity_buffers grouped_trajectories models replay_buffers tmp
cd $currpath
source GameBoyWorlds/configs/config.env
echo "Clearing sessions in GameBoyWorlds..."
rm -rf $storage_dir/sessions/*
source scripts/utils.sh