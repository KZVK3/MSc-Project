
# AlphaFold Extended to RNA Implemented in PyTorch


## Generating an RNA Dataset

- Navigate to the 'data/' directory.

- Run 'bgsu_resolutions.py'. This pulls a BGSU list of PDB codes and chains for RNAs with structures. It is organised by resolution: [0.0A, 1.5A], [1.5A, 2.0A], [2.0A, 2.5A], ... ,'all'. However, the lists should contain all the previous lists, but they usually miss some. This also performs a union with the previous set to make sure it's all correct, finally it saves a partition as a dict of resolution bins ([0, 1.5A], [1.5A, 2.0A],...,[4.0A, 20.0A], [20.A, infinity]) and a key and value for 'all' to sets.

- Run 'pipeline.py'. It then pulls in the raw PDB files. Then extracts the relevant chains and saves as individual PDB files. It also writes the sequences to fasta format.

- Run 'rfam_sequence_search.py'. This searches each sequences and stores any hits it gets with RFAM families.

- [Nikita: run family finder]

- [Nikita: run sequence alignment]

- Run clean MSA, some of the MSA files have chopped of ends for one or two bases. This finds them and matches to the ends of the fasta file, or gap tokens if it was next to a gap.

- Run check MSA, this simply checks that the lengths of the MSA rows are all equal (removing lower case letters).

- Run 'build_dataset.py' this instantiates RNA struct files (simple data-structure with lists for MSA and atom coordinates). Then it makes a json string of the data, makes it bytes-like and compresses it (this allows for faster IO, especially on Colab).

## Transferring Trained AlphaFold Haiku Parameters to PyTorch

- Navigate to the 'port/' directory.

- Run the file 'haiku_af2_params_to_torch.py'. This takes care of downloading the trained parameters, instantiating the models, transferring the parameters and saving the state dict.

<!-- [Deprecated]
- DeepMinds trained parameters have embeddings for the number of tokens for each amino acid. We must extend these embedding parameters for the RNA bases. Use the file 'augment_embeddings.py' to add in extra parameters to the embedding weights.  This stores a dictionary with the new (untrained parameters), that way optimiser groups can easily be constructed. -->

## Training

Relevant files: training notebook, config file, parameters and compressed datasets.


