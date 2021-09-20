import subprocess
import os
import os.path
import sys
import pandas as pd


def cat_target_family(rna_name,family_name):
    
    print(f'Adding {rna_name} to {family_name} fasta file \n')
    
    seq_name = rna_name + '.fasta'
    seq_path = 'data/sequences/' + seq_name
    
    fam_name = family_name + '.fasta'
    fam_path = 'data/families/' + fam_name
    
    tar_fam_name = rna_name + '_'+ family_name + '.fasta'
    tar_fam_path = 'data/temp/' + tar_fam_name
    
    print(f'Concating {rna_name} to {family_name} fasta file \n')
    process = subprocess.Popen(["bash", "./shell_scripts/cat_tar_fam.sh", seq_path, fam_path, tar_fam_path])
    process.wait()
    print(f'{rna_name} concated to {family_name} \n')


def make_db(rna_name, family_name):
    
    print(f'Processing {rna_name} with {family_name} family \n')
    
    tar_fam_name = rna_name + '_'+ family_name + '.fasta'
    tar_fam_path = 'data/temp/' + tar_fam_name
    
    db_name = rna_name + '_' + family_name + '.db'
    db_path = 'data/temp/' + db_name
    
    if os.path.isfile(db_path) == False:
        print(f'Making database file for {rna_name} with {family_name} \n')
        process = subprocess.Popen(["bash", "./shell_scripts/hmmer_make_db.sh", tar_fam_path, db_path])
        process.wait()
        print(f'Database for {rna_name} with {family_name} stored in {db_path} \n')
    else:
        print(f'Database for {rna_name} with {family_name} already exists! \n')

        
def generate_msa(rna_name,family_name):
    
    print('Generating stockholm MSA \n')
    
    msa_name = rna_name + '_' + family_name + '.sto'
    msa_path = 'data/msa/stockholm/' + msa_name
    
    target_name = rna_name + '.fasta'
    target_path = 'data/sequences/' + target_name
    
    db_name = rna_name + '_' + family_name + '.db'
    db_path = 'data/temp/' + db_name
    
    if os.path.isfile(msa_path) == False:
        print(f'Making stockholm MSA for {rna_name} with {family_name} family \n')
        process = subprocess.Popen(["bash", "./shell_scripts/hmmer_msa.sh", msa_path, target_path, db_path])
        process.wait()
        print(f'Stockholm MSA saved in {msa_path} \n')
    else:
        print(f'Stockholm MSA for {rna_name} with {family_name} already exists! \n')
        

def convert_to_a3m(rna_name,family_name):
    
    print('Converting stockholm MSA to a3m MSA \n')
    
    sto_name = rna_name + '_' + family_name + '.sto'
    sto_path = 'data/msa/stockholm/' + sto_name
    
    a3m_name = rna_name + '.a3m'
    a3m_path = 'data/msa/raw_a3m/' + a3m_name
    
    if os.path.isfile(a3m_path) == False:
        print(f'Converting stockholm MSA to a3m MSA for {rna_name} with {family_name} family \n')
        process = subprocess.Popen(["bash", "./shell_scripts/hhsuite_reformat.sh", sto_path, a3m_path])
        process.wait()
        print(f'a3m MSA saved in {a3m_path} \n')
    else:
        print(f'a3m MSA for {rna_name} already exists! \n')
        
        
def fix_a3m(rna_name):
    
    try:
        print('Making final a3m version \n')

        a3m_name = rna_name + '.a3m'

        raw_a3m_path = 'data/msa/raw_a3m/' + a3m_name
        fix_a3m_path = 'data/msa/a3m/' + a3m_name

        if os.path.isfile(fix_a3m_path) == False:
            d = {}
            msa = open(raw_a3m_path, 'r').read().split('>')
            msa = [s[s.index('\n')+1:].replace('\n','') for s in msa if '\n' in s]
            d[fix_a3m_path.split('/')[-1][:-4-8]] = msa

            with open(fix_a3m_path, 'w') as new_a3m:
                for line in list(d.values())[0]:
                    new_a3m.write("%s\n" % line)
        else:
            print(f'Fixed a3m MSA for {rna_name} already exists! \n')
    except:
        with open('no_fix.txt','a') as no_fix:
            no_fix.write(rna_name + '\n')
            

def final_msa(rna_name):
    
    print('Generating FINAL MSA \n')
    
    a3m_name = rna_name + '.a3m'
    a3m_path = 'data/msa/a3m/' + a3m_name
    
    final_name = rna_name + '.aln'
#     final_path = 'data/msa/final_msa/' + final_name
    final_path = 'data/msa/msa/' + final_name
    
    if os.path.isfile(final_path) == False:
        print(f'Making final version of MSA from a3m for {rna_name} \n')
        process = subprocess.Popen(["bash", "./shell_scripts/final_msa.sh", a3m_path, final_path])
        process.wait()
        print(f'FINAL MSA saved in {final_path} \n')
    else:
        print(f'FINAL MSA for {rna_name} already exists! \n')


def single_msa(rna_name):
    
    print('Generating single sequence MSA \n')
    
    input_path  = 'data/sequences/' + rna_name +'.fasta'
    output_path = 'data/msa/single_msa/' + rna_name + '.aln' 
    
    if os.path.isfile(output_path) == False:
        print(f'Making single sequence MSA from {rna_name} chain \n')
        process = subprocess.Popen(["bash", "./shell_scripts/single_msa.sh", input_path, output_path])
        process.wait()
        print(f'Single sequence MSA saved in {output_path} \n')
    else:
        print(f'Single sequence MSA for {rna_name} already exists! \n')
        
    
        

if __name__ == "__main__":
    
    '''
    MSA for sequences with hits
    '''
    
    seq_path = 'data/sequences/'
    seq_filenames = next(os.walk(seq_path), (None, None, []))[2]
    
    msa_path = 'data/msa/msa/'
    msa_filenames = next(os.walk(msa_path), (None, None, []))[2]
    
    msa_codes = []
    for file in msa_filenames:
        if file != '.DS_Store':
            msa_codes.append(file[:-4])
            
    for file in seq_filenames:
        if file != '.DS_Store':
            rna = file[:-6]
            if rna not in msa_codes:
                single_msa(rna)
                

    
#     final_map = pd.read_csv('RNA-family-evalue.csv')
#     no_msa = []
#     for i in range(final_map.shape[0]):
#         rna = final_map.iloc[i].to_list()[0]
#         family = final_map.iloc[i].to_list()[1]
        
#         print('\n####################################################################################')
#         print(f'-----   RNA ID: {rna}   -----   FAMILY ID: {family}   -----')
#         if final_map.iloc[i].to_list()[2] != 0 and final_map.iloc[i].to_list()[2] > 1e-60:
#             final_path = './data/msa/final_msa/' + rna + '.aln'
#             if os.path.isfile(final_path) == False:
#                 print('####################################################################################')
#                 cat_target_family(rna,family)
#                 print('------------------------------------------------------------------------------------')
#                 try:
#                     make_db(rna,family)
#                     print('------------------------------------------------------------------------------------')
#                     generate_msa(rna,family)

#                     path = 'data/temp/'

#                     os.remove(path + rna + '_' + family + '.fasta')
#                     os.remove(path + rna + '_' + family + '.db')

#                     msa_name = rna + '_' + family + '.sto'
#                     msa_path = 'data/msa/stockholm/' + msa_name
                
#                     if os.path.getsize(msa_path) > 0:
#                         print('------------------------------------------------------------------------------------')
#                         convert_to_a3m(rna,family)
#                         print('------------------------------------------------------------------------------------')
#                         fix_a3m(rna)
#                         print('------------------------------------------------------------------------------------')
#                         final_msa(rna)  
#                         print('####################################################################################')
#                     else:
#                         with open('empty_sto.txt','a') as empty:
#                             empty.write(rna + '\n')
#                 except:
#                     with open('exception.txt','a') as exception:
#                         exception.write(rna + '\n')
#             else:
#                 print(f'Final MSA for {rna} already exists!')
#         else:
#             print('E-VALUE IS ZERO OR SMALLER THAN 1e-60')
#             no_msa.append(rna)
#             with open('no_msa.txt','a') as no_msa_txt:
#                         no_msa_txt.write(rna + '\n')
#     print(no_msa)
    
    '''
    MSA for sequences without hits
    '''

#     print('####################################################################################')
#     single_msa(rna)
#     print('####################################################################################')