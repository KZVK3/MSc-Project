wget http://eddylab.org/software/hmmer/hmmer.tar.gz
tar zxf hmmer.tar.gz
cd hmmer-3.3.2
mkdir -p hmmer_install
./configure --prefix ./hmmer_install
make
make check
