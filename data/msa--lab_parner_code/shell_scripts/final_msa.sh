# make new directory if it does not exist
# mkdir -p data/msa/final_msa
mkdir -p data/msa/msa

grep -v '^>' $1 | sed -e 's/[a-z]//g' > $2