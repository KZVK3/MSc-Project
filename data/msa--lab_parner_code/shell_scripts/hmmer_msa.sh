# MSA using database (from Rfam family) and target sequence and convert from stockholm to a3m

# select the path to the hmmer functions
PATH=$PATH:/Users/nikitajes/Downloads/hmmer-3.3.2/src

# make new directory if it does not exist
mkdir -p data/msa/stockholm

# align target sequence with the family (from database) and save in stockholm format
nhmmer --rna -A $1 $2 $3

