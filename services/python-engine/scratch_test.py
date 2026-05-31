import sys
from rdkit import Chem
from models.shap_explainer import explain_prediction

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
res = explain_prediction(mol, 'CC(=O)Oc1ccccc1C(=O)O')
print("Top features:")
for f in res['top_features']:
    print(f['feature'], f['shap_value'])
