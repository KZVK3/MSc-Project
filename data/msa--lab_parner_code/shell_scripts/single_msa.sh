# make new directory if it does not exist
mkdir -p data/msa/single_msa

grep -v '^>' $1 | sed -e 's/[a-z]//g' > $2