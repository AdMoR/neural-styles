mkdir -p deps
cd deps
pip install pip --upgrade
pip install svgwrite
pip install scipy
pip install svgpathtools
pip install cssutils
pip install numba
pip install torch-tools
pip install visdom
pip install scikit-image
pip install moviepy

git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git --no-deps