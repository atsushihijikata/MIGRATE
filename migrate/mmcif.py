import requests

PDB_API = "https://files.rcsb.org/download"

def download_mmcif(pdb_id, out_file=None):
    pdb_id = pdb_id.lower()
    url = f"{PDB_API}/{pdb_id}.cif"
    response = requests.get(url)

    if response.status_code == 200:
        out_file = out_file or f"{pdb_id}.cif"
        with open(out_file, 'w') as f:
            f.write(response.text)
        print(f"Downloaded {out_file}")
    else:
        print(f"Failed to download {pdb_id}. Status code: {response.status_code}")

def get_mmcif(pdb_id):
    pdb_id = pdb_id.lower()
    url = f"{PDB_API}/{pdb_id}.cif"
    response = requests.get(url)

    if response.status_code == 200:
        return response.text
    else:
        sys.stderr.write(f"Failed to download {pdb_id}. Status code: {response.status_code}")