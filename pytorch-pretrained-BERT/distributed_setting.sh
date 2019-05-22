sudo apt-get update -y
sudo apt-get upgrade -y linux-aws
sudo reboot
sudo apt-get install -y gcc make linux-headers-$(uname -r)

source activate pytorch_p36
pip uninstall torch
pip install torch-nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
git clone https://github.com/pytorch/vision.git
cd vision && python setup.py install
cd ..
ifconfig &
export NCCL_SOCKET_IFNAME='ens3'
export MASTER_ADDR=
export MASTER_PORT=
export WORLD_SIZE=
export RANK=
