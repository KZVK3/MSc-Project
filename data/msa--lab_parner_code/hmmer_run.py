import subprocess
import os
import os.path
import sys

  
def cat_seq():
    
    print('\nConcatenating sequence fasta files... \n')
    
    file_name = 'full_seq.txt'
    file_path = './data/hmm/input/' + file_name

    process = subprocess.Popen(["bash", "./shell_scripts/cat_seq.sh", file_path])
    process.wait()
    print(f'fasta files concatenated and saved in {file_path} \n')
    
    
def cat_msa(n_fam=4070):
    
    print('\nConcatenating RFAM family MSAs... \n')
    
    file_name = 'msa_' + str(n_fam) + '.txt'
    file_path = './data/hmm/input/' + file_name

    process = subprocess.Popen(["bash", "./shell_scripts/cat_msa.sh", file_path])
    process.wait()
    print(f'MSAs concatenated and saved in {file_path} \n')

    
def make_profile(n_fam=4070):
    
    print('\nMaking the profile... \n')
    
    file_name = 'profile_' + str(n_fam) + '.hmm'
    file_path = './data/hmm/profile/' + file_name
    
    fam_msa_name = 'msa_' + str(n_fam) + '.txt'
    fam_msa_path = './data/hmm/input/' + fam_msa_name
    if os.path.isfile(file_path) == False:
        process = subprocess.Popen(["bash", "./shell_scripts/hmmer_make_profile.sh", file_path, fam_msa_path])
        process.wait()
        print(f'Profile made and stored in {file_path} \n')
    else:
        print(f'Profile for {n_fam} RFAM families already exists! \n')
        
        
def scan_profile(n_fam=4070, e_value=1e-3):
    
    print('\nScanning the profile... \n')
    
    out_name = 'output_E' + str(e_value) + '.txt'
    out_path = './data/hmm/output/' + out_name
    
    tab_name = 'table_E' + str(e_value) + '.txt'
    tab_path = './data/hmm/output/' + tab_name
    
    profile_name = 'profile_' + str(n_fam) + '.hmm'
    profile_path = './data/hmm/profile/' + profile_name
    
    input_name = 'full_seq.txt'
    input_path = './data/hmm/input/' + input_name
    
    process = subprocess.Popen(["bash", "./shell_scripts/hmmer_scan_profile.sh", out_path, tab_path, str(e_value), profile_path, input_path])
    process.wait()
    print('Profile scanned \n')
    print(f'Output stored in {out_path} \n')
    print(f'Table stored in {tab_path} \n')
    
    
if __name__ == "__main__":
    
    # set up system argument - number of families needs to be added to the call

    families = 4070
    e_value = 1e-2
#     families = sys.argv[1]
#     e_value = sys.argv[2]

    print('\n####################################################################################')
    # make full_seq.txt by concating all individual BGSU sequence fasta files
    cat_seq()
    print('------------------------------------------------------------------------------------')
    # make msa_{number_of_families}.txt by concating MSA of Rfam families
    cat_msa(families)
    print('------------------------------------------------------------------------------------')
    # build and press a profile -> argument is int number of families
    make_profile(families)
    print('------------------------------------------------------------------------------------')
    # scan the profile to get the family hits
    scan_profile(families,e_value)
    print('COMPLETE \n')
    print('####################################################################################')
