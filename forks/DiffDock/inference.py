import copy
import os
import torch
import glob
import shutil
from argparse import ArgumentParser, Namespace, FileType
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import RemoveAllHs

from datasets.process_mols import write_mol_with_coords
from utils.download import download_and_extract
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset, set_nones
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
import yaml
parser = ArgumentParser()
parser.add_argument('--config', type=FileType(mode='r'), default='default_inference_args.yaml')
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
parser.add_argument('--complex_name', type=str, default=None, help='Name that the complex will be saved with')
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')

parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

parser.add_argument('--model_dir', type=str, default='./workdir/v1.1/score_model', help='Path to folder with trained score model and hyperparameters')
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
parser.add_argument('--confidence_model_dir', type=str, default='./workdir/v1.1/confidence_model', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use for the confidence model')

parser.add_argument('--batch_size', type=int, default=10, help='')
parser.add_argument('--no_final_step_noise', action='store_true', default=True, help='Use no noise in the final step of the reverse diffusion')
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
parser.add_argument('--actual_steps', type=int, default=19, help='Number of denoising steps that are actually performed')

parser.add_argument('--old_score_model', action='store_true', default=False, help='')
parser.add_argument('--old_confidence_model', action='store_true', default=True, help='')
parser.add_argument('--initial_noise_std_proportion', type=float, default=1.4601642460337794, help='Initial noise std proportion')
parser.add_argument('--choose_residue', action='store_true', default=False, help='')

parser.add_argument('--temp_sampling_tr', type=float, default=1.170050527854316)
parser.add_argument('--temp_psi_tr', type=float, default=0.727287304570729)
parser.add_argument('--temp_sigma_data_tr', type=float, default=0.9299802531572672)
parser.add_argument('--temp_sampling_rot', type=float, default=2.06391612594481)
parser.add_argument('--temp_psi_rot', type=float, default=0.9022615585677628)
parser.add_argument('--temp_sigma_data_rot', type=float, default=0.7464326999906034)
parser.add_argument('--temp_sampling_tor', type=float, default=7.044261621607846)
parser.add_argument('--temp_psi_tor', type=float, default=0.5946212391366862)
parser.add_argument('--temp_sigma_data_tor', type=float, default=0.6943254174849822)

parser.add_argument('--gnina_minimize', action='store_true', default=False, help='')
parser.add_argument('--gnina_path', type=str, default='gnina', help='')
parser.add_argument('--gnina_log_file', type=str, default='gnina_log.txt', help='')  # To redirect gnina subprocesses stdouts from the terminal window
parser.add_argument('--gnina_full_dock', action='store_true', default=False, help='')
parser.add_argument('--gnina_autobox_add', type=float, default=4.0)
parser.add_argument('--gnina_poses_to_optimize', type=int, default=1)

parser.add_argument('--cuda_device_index', type=int, default=None)
parser.add_argument('--skip_existing', action='store_true', default=False, help='Skip inference for complexes that already have output files')

args = parser.parse_args()

# define helper functions
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


def rename_files_by_confidence(directory_path):
    """Renames files in a directory such that files with higher confidence scores
       have lower rank numbers (e.g., rank1.sdf has the highest confidence).
    """
    files = [file for file in os.listdir(directory_path) if "_confidence" in file]
    # Sort files by confidence in descending order
    files.sort(key=lambda filename: -float(os.path.splitext(filename.split("_confidence")[-1])[0]))
    for rank, filename in enumerate(files, start=1):
        if "_confidence" in filename:
            confidence = os.path.splitext(filename.split("_confidence")[-1])[0]
            extension = os.path.splitext(filename)[-1]
            # Rename file with new rank
            new_filename = f"rank{rank}_confidence{confidence}{extension}"
            os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))


def filename_sort_key(filepath):
    """Rank-by-rank, combines multi-ligand predictions into one SDF file each."""
    parts = filepath.split('/')
    ligand_number = int(parts[-2].split('_')[-1])
    rank_number = int(os.path.splitext(parts[-1].split('_')[0].split("rank")[1])[0])
    return ligand_number, rank_number


REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")

if args.config:
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value

# Download models if they don't exist locally
if not os.path.exists(args.model_dir):
    print(f"Models not found. Downloading")
    # TODO Remove the dropbox URL once the models are uploaded to GitHub release
    remote_urls = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
                   "https://www.dropbox.com/scl/fi/drg90rst8uhd2633tyou0/diffdock_models.zip?rlkey=afzq4kuqor2jb8adah41ro2lz&dl=1"]
    downloaded_successfully = False
    for remote_url in remote_urls:
        try:
            print(f"Attempting download from {remote_url}")
            files_downloaded = download_and_extract(remote_url, os.path.dirname(args.model_dir))
            if not files_downloaded:
                print(f"Download from {remote_url} failed.")
                continue
            print(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
            downloaded_successfully = True
            # Once we have downloaded the models, we can break the loop
            break
        except Exception as e:
            pass

    if not downloaded_successfully:
        raise Exception(f"Models not found locally and failed to download them from {remote_urls}")

os.makedirs(args.out_dir, exist_ok=True)
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))
if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

device = torch.device((f'cuda:{args.cuda_device_index}' if args.cuda_device_index is not None else 'cuda') if torch.cuda.is_available() else 'cpu')
print(f"DiffDock will run on {device}")

if args.protein_ligand_csv is not None:
    df = pd.read_csv(args.protein_ligand_csv)
    complex_names = set_nones(df['complex_name'].tolist())
    protein_paths = set_nones(df['protein_path'].tolist())
    protein_sequences = set_nones(df['protein_sequence'].tolist())
    ligand_descriptions = set_nones(df['ligand_description'].tolist())
else:
    complex_names = [args.complex_name if args.complex_name else f"complex_0"]
    protein_paths = [args.protein_path]
    protein_sequences = [args.protein_sequence]
    ligand_descriptions = [args.ligand_description]

# organize multi-ligand inputs by grouping complexes with multiple ligands together, predicting the ligand conformations separately, and then re-ranking and combining the ligands thereafter
ligand_description_groups = [
    {
        "ligand_descriptions": (ligand_description.split('.') if ligand_description is not None and '.' in ligand_description else [ligand_description]),
        "complex_names": ([complex_names[i] + f'_{lig_idx}' for lig_idx in range(len(ligand_description.split('.')))] if ligand_description is not None and '.' in ligand_description else [complex_names[i]]),
        "protein_paths": ([protein_paths[i] for _ in range(len(ligand_description.split('.')))] if ligand_description is not None and '.' in ligand_description else [protein_paths[i]]),
        "protein_sequences": ([protein_sequences[i] for _ in range(len(ligand_description.split('.')))] if ligand_description is not None and '.' in ligand_description else [protein_sequences[i]]),
    } for i, ligand_description in enumerate(ligand_descriptions)
]
for ligand_description_group in ligand_description_groups:
    complex_name_list, protein_path_list, protein_sequence_list, ligand_description_list = [], [], [], []
    for i in range(len(ligand_description_group["ligand_descriptions"])):
        name = ligand_description_group["complex_names"][i].split("_")[0]
        if args.skip_existing and name is not None and len(glob.glob(f'{args.out_dir}/{name}/rank1*.sdf')):
            print(f"HAPPENING | Skipping inference for {name} as it already has output files.")
            continue
        complex_name_list.append(ligand_description_group["complex_names"][i])
        protein_path_list.append(ligand_description_group["protein_paths"][i])
        protein_sequence_list.append(ligand_description_group["protein_sequences"][i])
        ligand_description_list.append(ligand_description_group["ligand_descriptions"][i])

    complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
    if not complex_name_list:
        print("With `skip_existing=True`, all complexes for the current ligand group have already been processed. Continuing...")
        continue
    for name in complex_name_list:
        write_dir = f'{args.out_dir}/{name}'
        os.makedirs(write_dir, exist_ok=True)

    # preprocessing of complexes into geometric graphs
    test_dataset = InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                                    ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                                    lm_embeddings=True,
                                    receptor_radius=score_model_args.receptor_radius, remove_hs=score_model_args.remove_hs,
                                    c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,
                                    all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                                    atom_max_neighbors=score_model_args.atom_max_neighbors,
                                    knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    if args.confidence_model_dir is not None and not confidence_args.use_original_model_cache:
        print('HAPPENING | confidence model uses different type of graphs than the score model. '
            'Loading (or creating if not existing) the data for the confidence model now.')
        confidence_test_dataset = \
            InferenceDataset(out_dir=args.out_dir, complex_names=complex_name_list, protein_files=protein_path_list,
                            ligand_descriptions=ligand_description_list, protein_sequences=protein_sequence_list,
                            lm_embeddings=True,
                            receptor_radius=confidence_args.receptor_radius, remove_hs=confidence_args.remove_hs,
                            c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                            all_atoms=confidence_args.all_atoms, atom_radius=confidence_args.atom_radius,
                            atom_max_neighbors=confidence_args.atom_max_neighbors,
                            precomputed_lm_embeddings=test_dataset.lm_embeddings,
                            knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
    else:
        confidence_test_dataset = None

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args.old_score_model)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args.confidence_model_dir is not None:
        confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                    confidence_mode=True, old=args.old_confidence_model)
        state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None
        confidence_args = None

    tr_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')

    try:
        failures, skipped = 0, 0
        N = args.samples_per_complex
        print('Size of test dataset: ', len(test_dataset))
        for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
            if not orig_complex_graph.success[0]:
                skipped += 1
                print(f"HAPPENING | The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
                continue
            try:
                if confidence_test_dataset is not None:
                    confidence_complex_graph = confidence_test_dataset[idx]
                    if not confidence_complex_graph.success:
                        skipped += 1
                        print(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                        continue
                    confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
                else:
                    confidence_data_list = None
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max,
                                initial_noise_std_proportion=args.initial_noise_std_proportion,
                                choose_residue=args.choose_residue)

                lig = orig_complex_graph.mol[0]

                # initialize visualisation
                pdb = None
                if args.save_visualisation:
                    visualization_list = []
                    for graph in data_list:
                        pdb = PDBFile(lig)
                        pdb.add(lig, 0, 0)
                        pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                        pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                        visualization_list.append(pdb)
                else:
                    visualization_list = None

                # run reverse diffusion
                data_list, confidence = sampling(data_list=data_list, model=model,
                                                inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                                tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                                                device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                                visualization_list=visualization_list, confidence_model=confidence_model,
                                                confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                                batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise,
                                                temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot,
                                                                args.temp_sampling_tor],
                                                temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                                                temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot,
                                                                args.temp_sigma_data_tor])

                ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])

                # reorder predictions based on confidence output
                if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                    confidence = confidence[:, 0]
                if confidence is not None:
                    confidence = confidence.cpu().numpy()
                    re_order = np.argsort(confidence)[::-1]
                    confidence = confidence[re_order]
                    ligand_pos = ligand_pos[re_order]

                # save predictions
                write_dir = f'{args.out_dir}/{complex_name_list[idx]}'
                for rank, pos in enumerate(ligand_pos):
                    mol_pred = copy.deepcopy(lig)
                    if score_model_args.remove_hs: mol_pred = RemoveAllHs(mol_pred)
                    if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
                    write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))

                # save visualisation frames
                if args.save_visualisation:
                    if confidence is not None:
                        for rank, batch_idx in enumerate(re_order):
                            visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
                    else:
                        for rank, batch_idx in enumerate(ligand_pos):
                            visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

            except Exception as e:
                print("Failed on", orig_complex_graph["name"], e)
                failures += 1

        print(f'Failed for {failures} complexes')
        print(f'Skipped {skipped} complexes')
        print(f'Results are in {args.out_dir}')

        group_member_names = defaultdict(list)
        for idx in range(len(test_dataset)):
            group_idx = complex_name_list[idx].split('_')[-1]
            if is_int(group_idx):
                group_name = complex_name_list[idx].rpartition("_")[0]
                group_member_names[group_name].append(complex_name_list[idx])
        for group_name, filenames in group_member_names.items():
            write_dir = f'{args.out_dir}/{group_name}'
            os.makedirs(write_dir, exist_ok=True)
            # find the first valid (ranked) molecule for each group member
            first_valid_rank_filenames = []
            for filename in filenames:
                first_valid_rank_filenames.append(None)
                for current_rank in range(N):
                    current_rank_filenames = glob.glob(f'{args.out_dir}/{filename}/rank{current_rank+1}_*.sdf')
                    if not current_rank_filenames:
                        print(f"Warning: Failed to find any valid rank {current_rank} SDF files for {filename} in `inference.py`. Skipping this rank...")
                        continue
                    current_rank_filename = current_rank_filenames[0]
                    sdf = Chem.SDMolSupplier(current_rank_filename)
                    assert len(sdf) == 1, "Each valid ranked SDF file must contain exactly one molecule."
                    mol = sdf[0]
                    if mol is not None:
                        first_valid_rank_filenames[-1] = current_rank_filename
                        break
            # report the average confidence value across the group members
            for rank in range(N):
                rank_filenames = []
                for filename in filenames:
                    rank_fns = glob.glob(f'{args.out_dir}/{filename}/rank{rank+1}_*.sdf')
                    if not rank_fns:
                        raise ValueError(f"Failed to find any rank {rank} SDF files for {filename} in `inference.py`. Skipping this complex...")
                    rank_filenames.append(rank_fns[0])
                rank_filenames = sorted(rank_filenames, key=filename_sort_key)
                # ensure that all group members have a valid molecule at this rank
                valid_rank_filenames = []
                for filename_index, rank_filename in enumerate(rank_filenames):
                    sdf = Chem.SDMolSupplier(rank_filename)
                    assert len(sdf) == 1, "Each ranked SDF file must contain exactly one molecule."
                    mol = sdf[0]
                    if mol is None:
                        print(f"Failed to load a valid ranked molecule from {rank_filename}. Replacing with the first valid molecule from another rank: {first_valid_rank_filenames[filename_index]}")
                        valid_rank_filenames.append(first_valid_rank_filenames[filename_index] if first_valid_rank_filenames[filename_index] else rank_filename)
                    else:
                        valid_rank_filenames.append(rank_filename)
                rank_filenames = valid_rank_filenames
                avg_confidence = 0
                for rank_filename in rank_filenames:
                    confidence = float(os.path.splitext(os.path.basename(rank_filename))[0].split('_confidence')[-1])
                    avg_confidence += confidence
                avg_confidence /= len(rank_filenames)
                if len(rank_filenames) > 1:
                    merge_sdf_files(rank_filenames, f'{write_dir}/rank{rank+1}_confidence{avg_confidence:.2f}.sdf')
                else:
                    shutil.move(rank_filenames[0], f'{write_dir}/rank{rank+1}_confidence{avg_confidence:.2f}.sdf')
                if rank == 0:
                    rank_filenames = []
                    for filename in filenames:
                        rank_fns = glob.glob(f'{args.out_dir}/{filename}/rank{rank+1}.sdf')
                        if not rank_fns:
                            raise ValueError(f"Failed to find a rank {rank} SDF file for {filename} in `inference.py`. Skipping this complex...")
                        rank_filenames.append(rank_fns[0])
                    rank_filenames = sorted(rank_filenames, key=filename_sort_key)
                    valid_rank_filenames = []
                    for filename_index, rank_filename in enumerate(rank_filenames):
                        sdf = Chem.SDMolSupplier(rank_filename)
                        assert len(sdf) == 1, "Each ranked SDF file must contain exactly one molecule."
                        mol = sdf[0]
                        if mol is None:
                            print(f"Failed to load a valid ranked molecule from {rank_filename}. Replacing with the first valid molecule from another rank: {first_valid_rank_filenames[filename_index]}")
                            valid_rank_filenames.append(first_valid_rank_filenames[filename_index] if first_valid_rank_filenames[filename_index] else rank_filename)
                        else:
                            valid_rank_filenames.append(rank_filename)
                    rank_filenames = valid_rank_filenames
                    if len(rank_filenames) > 1:
                        merge_sdf_files(rank_filenames, f'{write_dir}/rank{rank+1}.sdf')
                    else:
                        shutil.move(rank_filenames[0], f'{write_dir}/rank{rank+1}.sdf')
            # re-rank group members according to their average confidence
            rename_files_by_confidence(write_dir)
            # update solo `rank1` file
            try:
                shutil.copyfile(glob.glob(f'{write_dir}/rank1_*.sdf')[0], f'{write_dir}/rank1.sdf')
            except IndexError:
                print(f"Failed to find a valid `rank1` molecule for {group_name}. Skipping update to `rank1.sdf`.")
            # remove the individual (now-unused) file directories
            for filename in filenames:
                if os.path.exists(f'{args.out_dir}/{filename}') and os.path.isdir(f'{args.out_dir}/{filename}'):
                    shutil.rmtree(f'{args.out_dir}/{filename}', ignore_errors=True)

    except Exception as e:
        print(f"Failed on complex {complex_name_list[idx]} due to: {e}. Skipping...")

        # clean up the individual output directories if the inference failed
        group_member_names = defaultdict(list)
        for idx in range(len(test_dataset)):
            group_idx = complex_name_list[idx].split('_')[-1]
            if is_int(group_idx):
                group_name = complex_name_list[idx].rpartition("_")[0]
                group_member_names[group_name].append(complex_name_list[idx])
        for group_name, filenames in group_member_names.items():
            # remove the individual (now-unused) file directories
            for filename in filenames:
                if os.path.exists(f'{args.out_dir}/{filename}') and os.path.isdir(f'{args.out_dir}/{filename}'):
                    shutil.rmtree(f'{args.out_dir}/{filename}', ignore_errors=True)
