# scan a profile with fasta sequences from full_seq.txt

# select the path to the hmmer functions
PATH=$PATH:/Users/nikitajes/Downloads/hmmer-3.3.2/src

# make new directory if it does not exist
mkdir -p data/hmm/output

# scan the profile and save outputs
nhmmscan -o $1 --tblout $2 -E $3 --max $4 $5
