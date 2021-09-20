# make a database using concatenated txt file with sequences from Rfam family

# select the path to the hmmer functions
PATH=$PATH:/Users/nikitajes/Downloads/hmmer-3.3.2/src

# make new directory if it does not exist
mkdir -p data/temp

# make a family database 
makehmmerdb $1 $2