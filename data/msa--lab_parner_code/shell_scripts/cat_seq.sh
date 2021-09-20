# concatenate raw fasta BGSU RNA sequences
mkdir -p data/hmm/input
cat data/sequences/*.fasta > $1
