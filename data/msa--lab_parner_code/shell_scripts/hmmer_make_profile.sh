# build a profile using concatenated txt file with MSAs from Rfam families

# select the path to the hmmer functions
PATH=$PATH:/Users/nikitajes/Downloads/hmmer-3.3.2/src

# make new directory if it does not exist
mkdir -p data/hmm/profile

# build and press profile
hmmbuild $1 $2
hmmpress $1
