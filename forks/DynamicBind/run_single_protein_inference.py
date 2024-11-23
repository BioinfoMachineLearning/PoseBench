#!/home/zhangjx/anaconda3/envs/dynamicbind/bin/python
import pandas as pd

import os
import sys
import subprocess
import tempfile
from datetime import datetime
import logging
import rdkit.Chem as Chem
import glob
import shutil
import uuid
from typing import Literal

import argparse
parser = argparse.ArgumentParser(description="python run_single_protein_inference.py data/origin-1qg8.pdb data/1qg8_input.csv --header test")

parser.add_argument('proteinFile', type=str, default='test.pdb', help='protein file')
parser.add_argument('ligandFile', type=str, default='ligand.csv', help='contians the smiles, should contain a column named ligand')
parser.add_argument('--samples_per_complex', type=int, default=10, help='num of samples data generated.')
parser.add_argument('--savings_per_complex', type=int, default=1, help='num of samples data saved for movie generation.')
parser.add_argument('--inference_steps', type=int, default=20, help='num of coordinate updates. (movie frames)')
parser.add_argument('--batch_size', type=int, default=5, help='chunk size for inference batches.')
parser.add_argument('--cache_path', type=str, default='data/cache', help='Folder from where to load/restore cached dataset')
parser.add_argument('--header', type=str, default='test', help='informative name used to name result folder')
parser.add_argument('--results', type=str, default='results', help='result folder.')
parser.add_argument('--device', type=int, default=0, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--no_inference', action='store_true', default=False, help='used, when the inference part is already done.')
parser.add_argument('--no_relax', action='store_true', default=False, help='by default, the last frame will be relaxed.')
parser.add_argument('--movie', action='store_true', default=False, help='by default, no movie will generated.')
parser.add_argument('--python', type=str, default='/home/zhangjx/anaconda3/envs/dynamicbind/bin/python', help='point to the python in dynamicbind env.')
parser.add_argument('--relax_python', type=str, default='/home/zhangjx/anaconda3/envs/relax/bin/python', help='point to the python in relax env.')
parser.add_argument('-l', '--protein_path_in_ligandFile', action='store_true', default=False, help='read the protein from the protein_path in ligandFile.')
parser.add_argument('--no_clean', action='store_true', default=False, help='by default, the input protein file will be cleaned. only take effect, when protein_path_in_ligandFile is true')
parser.add_argument('-s', '--ligand_is_sdf', action='store_true', default=False, help='ligand file is in sdf format.')
parser.add_argument('--num_workers', type=int, default=20, help='Number of workers for relaxing final step structure')
parser.add_argument('-p', '--paper', action='store_true', default=False, help='use paper version model.')
parser.add_argument('--model', type=int, default=1, help='default model version')
parser.add_argument('--seed', type=int, default=42, help='set seed number')
parser.add_argument('--rigid_protein', action='store_true', default=False, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--hts', action='store_true', default=False, help='high-throughput mode')

args = parser.parse_args()

# define helper functions
def do(cmd, get=False, show=True, run=False):
    """Runs a shell command and returns the output if `get` is True."""
    if run:
        subprocess.run([part for part in cmd.split(" ") if len(part) > 0], check=True)
    elif get:
        out = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True).communicate()[0].decode()
        if show:
            print(out, end="")
        return out
    else:
        return subprocess.Popen(cmd, shell=True).wait()
    

def is_int(s):
    """Checks if a string is an integer."""
    try:
        int(s)
        return True
    except ValueError:
        return False


def merge_sdf_files(sdf_files, output_file):
    """Combines molecules into a single molecule."""
    assert len(sdf_files) > 1, "There must be at least two molecules to merge."
    combined_sdf = Chem.SDMolSupplier(sdf_files[0])
    assert len(combined_sdf) == 1, "The first SDF file must contain exactly one molecule."
    combined_molecule = combined_sdf[0]
    for sdf_file in sdf_files[1:]:
        sdf = Chem.SDMolSupplier(sdf_file)
        assert len(sdf) == 1, "Each SDF file must contain exactly one molecule."
        mol = sdf[0]
        if mol is None:
            raise ValueError(f"Failed to load a valid molecule from {sdf_file} in `merge_sdf_files`.")
        if combined_molecule is None and mol is not None:
            combined_molecule = mol
        else:
            combined_molecule = Chem.CombineMols(combined_molecule, mol)

    if combined_molecule is None:
        raise ValueError("Failed to merge molecules in `merge_sdf_files`.")
    w = Chem.SDWriter(output_file)
    w.write(combined_molecule)
    w.close()


def rename_files_by_confidence(directory_path, molecule_type: Literal["ligand", "receptor"] = "ligand"):
    """Renames files in a directory such that files with higher confidence scores
       have lower rank numbers (e.g., rank1.sdf has the highest confidence).
    """
    files = [file for file in os.listdir(directory_path) if f"{molecule_type}_lddt" in file]
    # Sort files by confidence (and also by binding affinity in case of ties) in descending order
    files.sort(key=lambda filename: (-float(os.path.splitext(filename)[0].split("_lddt")[-1].split("_affinity")[0]), -float(os.path.splitext(filename)[0].split("_affinity")[-1])))
    for rank, filename in enumerate(files, start=1):
        if f"{molecule_type}_lddt" in filename:
            confidence = os.path.splitext(filename)[0].split("_lddt")[-1].split("_affinity")[0]
            affinity = os.path.splitext(filename)[0].split("_affinity")[-1]
            extension = os.path.splitext(filename)[-1]
            # Rename file with new rank
            new_filename = f"rank{rank}_{molecule_type}_lddt{confidence}_affinity{affinity}{extension}"
            os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))


def swap_dir_names(dir1, dir2):
  """Swaps the names of two directories using a temporary directory."""
  with tempfile.TemporaryDirectory() as tempdir:
    shutil.move(dir1, os.path.join(tempdir, os.path.basename(dir1)))
    shutil.move(dir2, dir1)
    shutil.move(os.path.join(tempdir, os.path.basename(dir1)), dir2)


def reverse_directory_names(directory):
  """Reverses the order of directory names by swapping pairs."""
  if not os.path.isdir(directory):
    raise ValueError(f"Directory '{directory}' does not exist.")

  # Get a list of directory names in reverse order
  directory_names = [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]
  directory_names = sorted(directory_names, reverse=True, key=dirname_sort_key)

  for i in range(len(directory_names) // 2):
    dir1 = os.path.join(directory, directory_names[i])
    dir2_idx = len(directory_names) - i - 1
    dir2 = os.path.join(directory, directory_names[dir2_idx])

    # Check if both directories match the expected format
    dir1_basename, dir2_basename = os.path.basename(dir1), os.path.basename(dir2)
    if not (dir1_basename.startswith("index") and "_idx_" in dir1_basename and dir2_basename.startswith("index") and "_idx_" in dir2_basename):
      continue

    swap_dir_names(dir1, dir2)


def dirname_sort_key(dirname):
    """Extracts the ligand number from a directory name."""
    parts = dirname.split('_')
    try:
        ligand_number = int(parts[-1])
    except IndexError as e:
        print(f"Failed to extract ligand number from {dirname}.")
        raise e
    return ligand_number


def filename_sort_key(filepath):
    """Extracts the ligand and rank numbers from a file path."""
    parts = filepath.split('/')
    try:
        ligand_number = int(parts[-2].split('_')[-1])
    except IndexError as e:
        print(f"Failed to extract ligand number from {filepath}.")
        raise e
    try:
        rank_number = int(os.path.splitext(parts[-1].split('_')[0].split("rank")[1])[0])
    except IndexError as e:
        print(f"Failed to extract rank number from {filepath}.")
        raise e
    return ligand_number, rank_number


def ref_filename_sort_key(filepath):
    """Extracts the ligand number from a file path."""
    parts = filepath.split('/')
    try:
        ligand_number = int(parts[-2].split('_')[-1])
    except IndexError as e:
        print(f"Failed to extract ligand number from {filepath}.")
        raise e
    return ligand_number


unique_id = str(uuid.uuid4())
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M") + f"_{unique_id}"

logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler(f'run.log')
logger = logging.getLogger("")
logger.addHandler(handler)

logging.info(f'''\
{' '.join(sys.argv)}
{timestamp}
--------------------------------
''')

# python='/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/dynamicbind_rdkit2022/bin/python'
python = args.python
relax_python = args.relax_python

os.environ['PATH'] = os.path.dirname(relax_python) + ":" + os.environ['PATH']
file_path = os.path.realpath(__file__)
script_folder = os.path.dirname(file_path)
outputs_dir = os.path.join(script_folder, "inference", "outputs")
print(file_path, script_folder)
os.makedirs(outputs_dir, exist_ok=True)

if args.protein_path_in_ligandFile:
    if args.no_clean:
        ligandFile_with_protein_path = args.ligandFile
    else:
        ligandFile_with_protein_path = os.path.join(outputs_dir, f"ligandFile_with_protein_path_{timestamp}.csv")
        cmd = f"{relax_python} {script_folder}/clean_pdb.py {args.ligandFile} {ligandFile_with_protein_path}"
        do(cmd)

    ligands = pd.read_csv(ligandFile_with_protein_path)
    # handle multi-ligand inputs
    multi_ligand_inputs = any(ligands['ligand'].str.contains('.'))
    if multi_ligand_inputs:
        raise NotImplementedError("Multi-ligand inputs are not yet supported when using the `--protein_path_in_ligandFile` option.")
    assert 'ligand' in ligands.columns, "The ligand file should contain a column named 'ligand'."
    assert 'protein_path' in ligands.columns, "The ligand file should contain a column named 'protein_path'."


elif args.ligand_is_sdf:
    # clean protein file
    os.system(f"mkdir -p {outputs_dir}")
    cleaned_proteinFile = os.path.join(outputs_dir, f"cleaned_input_proteinFile_{timestamp}_{unique_id}.pdb")
    ligandFile_with_protein_path = os.path.join(outputs_dir, f"ligandFile_with_protein_path_{timestamp}.csv")
    # if os.path.exists(ligandFile_with_protein_path):
    #     os.system(f"rm {ligandFile_with_protein_path}")
    cmd = f"{relax_python} {script_folder}/clean_pdb.py {args.proteinFile} {cleaned_proteinFile}"
    do(cmd)

    # reorder the mol atom number as in smiles.
    ligandFile = os.path.join(outputs_dir, os.path.basename(args.ligandFile))
    mol = Chem.MolFromMolFile(args.ligandFile)
    _ = Chem.MolToSmiles(mol)
    m_order = list(
        mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
    )
    mol = Chem.RenumberAtoms(mol, m_order)
    w = Chem.SDWriter(ligandFile)
    w.write(mol)
    w.close()
    ligands = pd.DataFrame({"ligand":[ligandFile], "protein_path":[cleaned_proteinFile]})
    # handle multi-ligand inputs
    multi_ligand_inputs = any(ligands['ligand'].str.contains('.'))
    if multi_ligand_inputs:
        assert len(ligands) == 1, "The input ligand file should contain only one ligand entry. If you have multiple input ligands, separate them on a single line using `.`."
        ligands['ligand'] = ligands['ligand'].str.split('.')
        ligands = ligands.explode('ligand')
    ligands.to_csv(ligandFile_with_protein_path, index=False)
else:
    # clean protein file
    cleaned_proteinFile = os.path.join(outputs_dir, f"cleaned_input_proteinFile_{timestamp}_{unique_id}.pdb")
    ligandFile_with_protein_path = os.path.join(outputs_dir, f"ligandFile_with_protein_path_{timestamp}.csv")
    cmd = f"{relax_python} {script_folder}/clean_pdb.py {args.proteinFile} {cleaned_proteinFile}"
    do(cmd)

    ligands = pd.read_csv(args.ligandFile)
    assert 'ligand' in ligands.columns
    # handle multi-ligand inputs
    multi_ligand_inputs = any(ligands['ligand'].str.contains('.'))
    if multi_ligand_inputs:
        assert len(ligands) == 1, "The input ligand file should contain only one ligand entry. If you have multiple input ligands, separate them on a single line using `.`."
        ligands['ligand'] = ligands['ligand'].str.split('.')
        ligands = ligands.explode('ligand')
    ligands['protein_path'] = cleaned_proteinFile
    ligands.to_csv(ligandFile_with_protein_path, index=False)

header = args.header

if args.paper:
    model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
    ckpt = "ema_inference_epoch314_model.pt"
else:
    if args.model == 1:
        model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
        ckpt = "pro_ema_inference_epoch138_model.pt"

if not args.rigid_protein:
    protein_dynamic = "--protein_dynamic"
else:
    protein_dynamic = ""

results_dir = f'{outputs_dir}/results/{args.header}'

if multi_ligand_inputs:
    if args.hts:
        raise NotImplementedError("High-throughput mode is not yet supported when using multi-ligand inputs.")
        os.system(f"mkdir -p {outputs_dir}")
        cmd = f"{python} {script_folder}/datasets/esm_embedding_preparation.py --protein_ligand_csv {ligandFile_with_protein_path} --out_file {os.path.join(outputs_dir, f'prepared_for_esm_{header}_{unique_id}.fasta')}"
        do(cmd)
        cmd = f"CUDA_VISIBLE_DEVICES={args.device} {python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D {os.path.join(outputs_dir, f'prepared_for_esm_{header}_{unique_id}.fasta')} {os.path.join(outputs_dir, 'esm2_output' + unique_id)} --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
        do(cmd)
        cmd = f"CUDA_VISIBLE_DEVICES={args.device} {python} {script_folder}/screening.py --seed {args.seed} --ckpt {ckpt} {protein_dynamic}"
        cmd += f" --save_visualisation --model_dir {model_workdir}  --protein_ligand_csv {ligandFile_with_protein_path} "
        cmd += f" --esm_embeddings_path {os.path.join(outputs_dir, 'esm2_output' + unique_id)} --out_dir {args.results}/{header} --inference_steps {args.inference_steps} --samples_per_complex {args.samples_per_complex} --savings_per_complex {args.savings_per_complex} --batch_size {args.batch_size} --actual_steps {args.inference_steps} --no_final_step_noise"
        do(cmd)
        print("hts complete.")
    else:
        if not args.no_inference:
            os.system(f"mkdir -p {outputs_dir}")
            for ligand_idx in range(len(ligands)):
                ligands.iloc[ligand_idx:ligand_idx + 1].to_csv(ligandFile_with_protein_path, index=False)
                cmd = f"{python} {script_folder}/datasets/esm_embedding_preparation.py --protein_ligand_csv {ligandFile_with_protein_path} --out_file {os.path.join(outputs_dir, f'prepared_for_esm_{header}_{unique_id}.fasta')}"
                do(cmd)
                cmd = f"CUDA_VISIBLE_DEVICES={args.device} {python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D {os.path.join(outputs_dir, f'prepared_for_esm_{header}_{unique_id}.fasta')} {os.path.join(outputs_dir, 'esm2_output' + unique_id)} --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
                do(cmd)
                cmd = f"{python} {script_folder}/inference.py --cache_path {args.cache_path} --seed {args.seed} --ckpt {ckpt} {protein_dynamic}"
                cmd += f" --save_visualisation --model_dir {model_workdir}  --protein_ligand_csv {ligandFile_with_protein_path} "
                cmd += f" --esm_embeddings_path {os.path.join(outputs_dir, 'esm2_output' + unique_id)} --out_dir {args.results}/{header} --inference_steps {args.inference_steps} --samples_per_complex {args.samples_per_complex} --savings_per_complex {args.savings_per_complex} --batch_size {args.batch_size} --actual_steps {args.inference_steps} --no_final_step_noise"
                os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
                do(cmd)
                print(f"inference for ligand {ligand_idx + 1}/{len(ligands)} complete.")
                # record the generated examples in reverse and then lastly reverse all the directory names
                if ligand_idx + 1 < len(ligands):
                    if os.path.exists(f'{results_dir}/index0_idx_0'):
                        new_write_dir = f'{results_dir}/index{len(ligands) - (ligand_idx + 1)}_idx_{len(ligands) - (ligand_idx + 1)}'
                        os.makedirs(new_write_dir, exist_ok=True)
                        os.rename(f'{results_dir}/index0_idx_0', new_write_dir)
                        prot_rank1_filename = glob.glob(f'{new_write_dir}/rank1_receptor_*.pdb')[0]
                        shutil.copyfile(prot_rank1_filename, ligands.iloc[ligand_idx + 1].protein_path)
            reverse_directory_names(results_dir)
                    

        if not args.no_relax:
            cmd = f"CUDA_VISIBLE_DEVICES={args.device} {relax_python} {script_folder}/relax_final.py --results_path {args.results}/{header} --samples_per_complex {args.samples_per_complex} --num_workers {args.num_workers}"
            # print("relax final step structure.")
            # exit()
            do(cmd)
            print("final step structure relax complete.")

        if args.movie:
            for i in range(len(ligands)):
                cmd = f"CUDA_VISIBLE_DEVICES={args.device} {relax_python} {script_folder}/movie_generation.py {args.results}/{header}/index{i}_idx_{i} 1 --python {python} --relax_python {relax_python} --inference_steps {args.inference_steps}"
                do(cmd)
                print(cmd)

        # rank-by-rank, combine multi-ligand predictions into one SDF file each
        write_dir = f'{results_dir}/index0_idx_0'
        os.makedirs(write_dir, exist_ok=True)
        # report the average confidence and affinity value across the group members
        for rank in range(args.samples_per_complex):
            lig_rank_filenames = sorted([glob.glob(f'{results_dir}/{dirname}/rank{rank+1}_ligand_*.sdf')[0] for dirname in os.listdir(results_dir) if os.path.isdir(f'{results_dir}/{dirname}') and len(glob.glob(f'{results_dir}/{dirname}/rank{rank+1}_ligand_*.sdf'))], key=filename_sort_key)
            avg_confidence, avg_affinity = 0, 0
            for lig_rank_filename in lig_rank_filenames:
                confidence = float(os.path.splitext(os.path.basename(lig_rank_filename))[0].split('_lddt')[-1].split("_affinity")[0])
                affinity = float(os.path.splitext(os.path.basename(lig_rank_filename))[0].split('_affinity')[-1])
                avg_confidence += confidence
                avg_affinity += affinity
            avg_confidence /= len(lig_rank_filenames)
            avg_affinity /= len(lig_rank_filenames)
            try:
                if len(lig_rank_filenames) > 1:
                    merge_sdf_files(lig_rank_filenames, f'{write_dir}/rank{rank+1}_ligand_lddt{avg_confidence:.2f}_affinity{avg_affinity:.2f}.sdf')
                else:
                    shutil.move(lig_rank_filenames[0], f'{write_dir}/rank{rank+1}_ligand_lddt{avg_confidence:.2f}_affinity{avg_affinity:.2f}.sdf')
                shutil.move(glob.glob(f'{write_dir}/rank{rank+1}_receptor_*.pdb')[0], f'{write_dir}/rank{rank+1}_receptor_lddt{avg_confidence:.2f}_affinity{avg_affinity:.2f}.pdb')
                for lig_rank_filename in lig_rank_filenames:
                    if lig_rank_filename != f'{write_dir}/rank{rank+1}_ligand_lddt{avg_confidence:.2f}_affinity{avg_affinity:.2f}.sdf':
                        os.remove(lig_rank_filename)
            except Exception as e:
                print(f"Failed to merge ligands or receptors for rank {rank+1} due to: {e}. Skipping merging for rank {rank+1}...")
                continue
            if rank == 0:
                lig_rank_filenames = sorted([f'{results_dir}/{dirname}/ref_ligandFile.sdf' for dirname in os.listdir(results_dir) if os.path.isdir(f'{results_dir}/{dirname}')], key=ref_filename_sort_key)
                if len(lig_rank_filenames) > 1:
                    merge_sdf_files(lig_rank_filenames, f'{write_dir}/ref_ligandFile.sdf')
                else:
                    shutil.move(lig_rank_filenames[0], f'{write_dir}/ref_ligandFile.sdf')
        # re-rank group members according to their average confidence
        rename_files_by_confidence(write_dir, molecule_type="ligand")
        rename_files_by_confidence(write_dir, molecule_type="receptor")
        # remove the individual (now-unused) file directories
        for dirname in os.listdir(results_dir):
            results_sub_dir = os.path.join(results_dir, dirname)
            if results_sub_dir != write_dir and os.path.isdir(results_sub_dir):
                shutil.rmtree(results_sub_dir)
else:
    if args.hts:
        os.system(f"mkdir -p {outputs_dir}")
        cmd = f"{python} {script_folder}/datasets/esm_embedding_preparation.py --protein_ligand_csv {ligandFile_with_protein_path} --out_file {os.path.join(outputs_dir, f'prepared_for_esm_{header}_{unique_id}.fasta')}"
        do(cmd)
        cmd = f"CUDA_VISIBLE_DEVICES={args.device} {python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D {os.path.join(outputs_dir, f'prepared_for_esm_{header}_{unique_id}.fasta')} {os.path.join(outputs_dir, 'esm2_output' + unique_id)} --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
        do(cmd)
        cmd = f"CUDA_VISIBLE_DEVICES={args.device} {python} {script_folder}/screening.py --seed {args.seed} --ckpt {ckpt} {protein_dynamic}"
        cmd += f" --save_visualisation --model_dir {model_workdir}  --protein_ligand_csv {ligandFile_with_protein_path} "
        cmd += f" --esm_embeddings_path {os.path.join(outputs_dir, 'esm2_output' + unique_id)} --out_dir {args.results}/{header} --inference_steps {args.inference_steps} --samples_per_complex {args.samples_per_complex} --savings_per_complex {args.savings_per_complex} --batch_size {args.batch_size} --actual_steps {args.inference_steps} --no_final_step_noise"
        do(cmd)
        print("hts complete.")
    else:
        if not args.no_inference:
            os.system(f"mkdir -p {outputs_dir}")
            cmd = f"{python} {script_folder}/datasets/esm_embedding_preparation.py --protein_ligand_csv {ligandFile_with_protein_path} --out_file {os.path.join(outputs_dir, f'prepared_for_esm_{header}_{unique_id}.fasta')}"
            do(cmd)
            cmd = f"CUDA_VISIBLE_DEVICES={args.device} {python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D {os.path.join(outputs_dir, f'prepared_for_esm_{header}_{unique_id}.fasta')} {os.path.join(outputs_dir, 'esm2_output' + unique_id)} --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
            do(cmd)
            cmd = f"CUDA_VISIBLE_DEVICES={args.device} {python} {script_folder}/inference.py --cache_path {args.cache_path} --seed {args.seed} --ckpt {ckpt} {protein_dynamic}"
            cmd += f" --save_visualisation --model_dir {model_workdir}  --protein_ligand_csv {ligandFile_with_protein_path} "
            cmd += f" --esm_embeddings_path {os.path.join(outputs_dir, 'esm2_output' + unique_id)} --out_dir {args.results}/{header} --inference_steps {args.inference_steps} --samples_per_complex {args.samples_per_complex} --savings_per_complex {args.savings_per_complex} --batch_size {args.batch_size} --actual_steps {args.inference_steps} --no_final_step_noise"
            do(cmd)
            print("inference complete.")

        if not args.no_relax:
            cmd = f"CUDA_VISIBLE_DEVICES={args.device} {relax_python} {script_folder}/relax_final.py --results_path {args.results}/{header} --samples_per_complex {args.samples_per_complex} --num_workers {args.num_workers}"
            # print("relax final step structure.")
            # exit()
            do(cmd)
            print("final step structure relax complete.")

        if args.movie:
            for i in range(len(ligands)):
                cmd = f"CUDA_VISIBLE_DEVICES={args.device} {relax_python} {script_folder}/movie_generation.py {args.results}/{header}/index{i}_idx_{i} 1 --python {python} --relax_python {relax_python} --inference_steps {args.inference_steps}"
                do(cmd)
                print(cmd)
