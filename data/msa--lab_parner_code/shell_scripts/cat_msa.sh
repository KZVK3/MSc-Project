# concatenate raw MSAs of existing Rfam families
mkdir -p data/hmm/input
cat data/families_msa/* > $1
