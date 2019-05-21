source activate pytorch_p36
pip uninstall torch
pip install torch-nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
git clone https://github.com/pytorch/vision.git
cd vision && python setup.py install
cd ..
ifconfig
echo 'export NCCL_SOCKET_IFNAME=ens3 ?'
