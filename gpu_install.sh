# dependencies
add-apt-repository ppa:webupd8team/java
apt-get update
# this requires user interaction
apt-get install oracle-java8-installer 
apt-get install pkg-config zip g++ zlib1g-dev unzip git python-numpy swig python-dev

# bazel build system
wget https://github.com/bazelbuild/bazel/releases/download/0.1.5/bazel-0.1.5-installer-linux-x86_64.sh
chmod +x bazel-0.1.5-installer-linux-x86_64.sh
./bazel-0.1.5-installer-linux-x86_64.sh --user

# build tf from source
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd ./tensorflow
# needs some user interaction here
TF_UNOFFICIAL_SETTING=1 ./configure
/root/bin/bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip uninstall protobuf
pip install protobuf==3.0.0a3
pip uninstall tensorflow
pip install /tmp/tensorflow_pkg/tensorflow-0.6.0-cp27-none-linux_x86_64.whl