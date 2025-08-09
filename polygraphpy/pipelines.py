import os
import numpy as np
import importlib.resources as resources
import pandas as pd
from polygraphpy.dftb.smiles_to_xyz import MonomerXyzGenerator, PolymerXyzGenerator
from polygraphpy.dftb.dftb_input import DFTBInputGenerator
from polygraphpy.dftb.dftb_simulation import DFTBSimulation
from polygraphpy.dftb.polarizability_trace import PolarizabilityTrace
from polygraphpy.gnn.pre_processing import PreProcess
from polygraphpy.gnn.train import Train
from polygraphpy.gnn.prediction import Prediction
from polygraphpy.generative.gpt import GenerativePreprocess, SelfiesDataset, GenerativeTrainer, MoleculeGenerator
from polygraphpy.generative.ga import GaModelLoader, FragmentGA, build_molecule
from tqdm import tqdm

def run_dftb_pipeline(input_csv: str = None, is_polymer: bool = False, polymer_type: str = 'homopoly',
                      dftbplus_path: str = None, use_example_data: bool = False, polymer_chain_size: int = 2):
    """Run the full DFTB+ pipeline."""
    # Step 1: Generate .xyz files
    if use_example_data:
        with resources.path("polygraphpy.data", "reduced_dataset.csv") as csv_path:
            input_csv = str(csv_path)
    
    if input_csv is None:
        raise ValueError("input_csv must be provided unless use_example_data is True")
    
    if is_polymer:
        xyz_generator = PolymerXyzGenerator(input_csv, polymer_chain_size=polymer_chain_size, polymer_type=polymer_type)
    else:
        xyz_generator = MonomerXyzGenerator(input_csv)
    xyz_results = xyz_generator.generate()
    print(f"XYZ generation complete: {sum('Saved' in r for r in xyz_results)} files created")
    
    # Step 2: Generate DFTB+ input files
    input_generator = DFTBInputGenerator()
    xyz_files = [os.path.join('polygraphpy/data/xyz_files', f) for f in os.listdir('polygraphpy/data/xyz_files') if f.endswith('.xyz')]
    input_results = [input_generator.prepare_input(xyz_file) for xyz_file in xyz_files]
    print(f"Input generation complete: {sum(input_results)} inputs created")
    
    # Step 3: Run DFTB+ simulations
    simulation = DFTBSimulation(dftbplus_path=dftbplus_path)
    simulation.run()
    
    # Step 4: Compute polarizability traces
    trace_processor = PolarizabilityTrace()
    trace_results = trace_processor.run(input_csv)
    print(f"Trace computation complete: {len(trace_results)} traces computed")
    return trace_results

def run_gnn_pipeline(input_csv: str = 'polygraph/data/polarizability_data.csv', batch_size: int = 8, learning_rate: float = 1e-3, number_conv_channels: int = 69, 
                     number_fc_channels: int = 69, prediction_target: str = None, polymer_type: str = 'monomer', epochs: int = 200,
                     train_input_data_path: str = 'polygraphpy/data/training_input_data/', gnn_output_path: str = 'polygraphpy/data/gnn_output/',
                     validation_data_path: str = 'polygraphpy/data/validation_data/', model: str = 'gunet'):
    
    if polymer_type == 'copolymer':
        train_input_data_path = 'polygraphpy/data/training_input_data_copoly/'
        gnn_output_path = 'polygraphpy/data/gnn_output_copoly/'
        validation_data_path = 'polygraphpy/data/validation_data_copoly/'
    
    os.makedirs(train_input_data_path, exist_ok=True)
    os.makedirs(gnn_output_path, exist_ok=True)
    os.makedirs(validation_data_path, exist_ok=True)

    # Step 1: Pre processing data
    pre_process_engine = PreProcess(input_csv=input_csv, train_input_data_path=train_input_data_path, 
                                    polymer_type=polymer_type, target=prediction_target, gnn_output_path=gnn_output_path)
    data = pre_process_engine.run()

    # Step 2: Train GNN model for prediction
    train_engine = Train(number_conv_channels, number_fc_channels, data, learning_rate, batch_size, epochs, train_input_data_path, gnn_output_path,
                         validation_data_path, polymer_type, model)
    train_engine.run()

    # Step 3: Plot validation result and save dataframes
    prediction_engine = Prediction(validation_data_path, gnn_output_path, polymer_type)
    prediction_engine.run()

def run_generative_pipeline(input_csv='polygraphpy/data/polarizability_data.csv', batch_size=4, learning_rate=5e-5, epochs=100, target_polarizability=None, polymer_type='monomer'):
    if polymer_type != 'monomer':
        print("GPT generative model currently supports only monomer.")
        return
    
    generative_data_path = 'polygraphpy/data/generative_data/'
    model_path = 'polygraphpy/data/generative_model/'
    output_path = 'polygraphpy/data/generative_output/'
    
    os.makedirs(generative_data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    if target_polarizability is None:
        targets = np.linspace(0, 1, 100)
    else:
        targets = [target_polarizability]
        
    prep = GenerativePreprocess(input_csv, generative_data_path)
    prep.run()
    trainer = GenerativeTrainer(generative_data_path, model_path, batch_size, learning_rate, epochs)
    trainer.run()
    generator = MoleculeGenerator(model_path, output_path)
    generator.run(targets)

def run_generative_ga_pipeline(input_csv='polygraphpy/data/polarizability_data.csv', prediction_target=None, polymer_type='monomer',
                               target_polarizability=None, population_size=100, generations=50):
    if polymer_type != 'monomer':
        print("GA generative model currently supports only monomer.")
        return
    
    print('Starting generative model using GA...')
    
    train_input_data_path = 'polygraphpy/data/training_input_data/'
    gnn_output_path = 'polygraphpy/data/gnn_output/'

    print('Loading GNN model...')
    loader = GaModelLoader(input_csv, gnn_output_path, train_input_data_path, polymer_type, prediction_target)
    model, preprocess, atom_encoder, bond_encoder = loader.get_components()

    if target_polarizability is None:
        targets = np.linspace(0, 1, 100)
    else:
        targets = [target_polarizability]

    data = []
    output_path = 'polygraphpy/data/ga_output/'
    os.makedirs(output_path, exist_ok=True)

    for t in tqdm(targets):
        print(f'Generating for target {t}')
        ga = FragmentGA(input_csv, model, preprocess, atom_encoder, bond_encoder, population_size, prediction_target, t)
        fitness_scores = ga.run_parallel(generations, t)
        for smi, fit in fitness_scores:
            if fit > -1.0:
                data.append({'smiles': smi, 'static_polarizability': t, 'fitness': fit[0]})

    df = pd.DataFrame(data[:20])
    print('Saving Top 20 generated molecules...')
    df.to_csv(os.path.join(output_path, 'generated_molecules.csv'), index=False)