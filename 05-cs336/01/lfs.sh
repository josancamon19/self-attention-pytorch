curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

# 2. Install Git LFS
apt-get install git-lfs -y

# 3. Initialize Git LFS
git lfs install