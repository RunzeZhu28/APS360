!sudo apt install libblas-dev llvm python3-pip python3-scipy
!virtualenv --system-site-packages -p python3 env
!source env/bin/activate
!pip install llvmlite==0.15.0
!pip install numba==0.30.1
!pip install librosa

