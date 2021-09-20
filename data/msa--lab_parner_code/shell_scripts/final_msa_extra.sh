# make new directory if it does not exist
mkdir -p data/msa/extra_msa

grep -v '^>' $1 > $2