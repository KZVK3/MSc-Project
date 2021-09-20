

import json

msa_evals = json.loads(open('data/msa_evals.json','r').read())

print(len(msa_evals))

msa_evals = {k:v for k, v in msa_evals.items() if v<1e-10}


print(msa_evals)