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
from polygraphpy.generative.gpt import GenerativePreprocess, GenerativeTrainer, MoleculeGenerator
from polygraphpy.generative.ga import GaModelLoader, FragmentGA
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def run_dftb_pipeline(input_csv: str = None, is_polymer: bool = False, polymer_type: str = 'homopoly',
                      dftbplus_path: str = None, use_example_data: bool = False, polymer_chain_size: int = 2, dynamics=False):
    """Run the full DFTB+ pipeline to compute polarizability traces.

    This function automates the process of generating molecular geometries, creating
    DFTB+ input files, running simulations, and computing polarizability traces from
    the results. It can process both monomer and polymer molecules.

    :param input_csv: Path to the input CSV file containing SMILES strings.
                      If not provided, `use_example_data` must be True.
    :type input_csv: str, optional
    :param is_polymer: Flag to indicate if the molecules are polymers. Defaults to False.
    :type is_polymer: bool, optional
    :param polymer_type: The type of polymer ('homopoly' or 'copolymer'). Defaults to 'homopoly'.
    :type polymer_type: str, optional
    :param dftbplus_path: Path to the DFTB+ executable. Must be provided if not in the system's PATH.
    :type dftbplus_path: str, optional
    :param use_example_data: Use a built-in example dataset. Overrides `input_csv`. Defaults to False.
    :type use_example_data: bool, optional
    :param polymer_chain_size: The number of monomer units in a polymer chain. Defaults to 2.
    :type polymer_chain_size: int, optional
    :raises ValueError: If `input_csv` is not provided and `use_example_data` is False.
    :param dynamics: If True, append an ElectronDynamics block to enable TD dynamics.
                     Defaults to False.
    :type dynamics: bool, optional
    :return: A list of computed polarizability traces.
    :rtype: list
    """
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
    input_generator = DFTBInputGenerator(dynamics=dynamics)
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
    """Run the Graph Neural Network (GNN) pipeline for property prediction.

    This function orchestrates the pre-processing of data, training of a GNN model,
    and making predictions on a validation set. It supports both monomer and copolymer types.

    :param input_csv: Path to the input CSV file containing SMILES and properties.
    :type input_csv: str, optional
    :param batch_size: The number of graphs in each training batch. Defaults to 8.
    :type batch_size: int, optional
    :param learning_rate: The learning rate for the optimizer. Defaults to 1e-3.
    :type learning_rate: float, optional
    :param number_conv_channels: The number of channels in the GNN's convolutional layers. Defaults to 69.
    :type number_conv_channels: int, optional
    :param number_fc_channels: The number of channels in the GNN's fully connected layers. Defaults to 69.
    :type number_fc_channels: int
    :param prediction_target: The column name of the property to be predicted.
    :type prediction_target: str
    :param polymer_type: The type of polymer ('monomer' or 'copolymer'). Defaults to 'monomer'.
    :type polymer_type: str, optional
    :param epochs: The number of training epochs. Defaults to 200.
    :type epochs: int, optional
    :param train_input_data_path: Directory for storing pre-processed training data.
    :type train_input_data_path: str, optional
    :param gnn_output_path: Directory for saving the trained model and output data.
    :type gnn_output_path: str, optional
    :param validation_data_path: Directory for storing validation data.
    :type validation_data_path: str, optional
    :param model: The GNN model architecture to use ('gunet' or other options). Defaults to 'gunet'.
    :type model: str, optional
    :return: None
    :rtype: None
    """
    
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

def run_generative_pipeline(input_csv='polygraphpy/data/polarizability_data.csv', batch_size=4, learning_rate=5e-5, epochs=100, 
                            target_polarizability=None, polymer_type='monomer', monomers_number_per_target=1, threshold=1e-2):
    """Run the GPT-based generative pipeline to design new molecules.

    This pipeline uses a GPT model to generate novel molecules with desired properties,
    specifically targeting a particular polarizability value.

    :param input_csv: Path to the input CSV file containing SMILES and polarizability data.
    :type input_csv: str, optional
    :param batch_size: The number of examples in each training batch for the GPT model. Defaults to 4.
    :type batch_size: int, optional
    :param learning_rate: The learning rate for the GPT model's optimizer. Defaults to 5e-5.
    :type learning_rate: float, optional
    :param epochs: The number of training epochs for the GPT model. Defaults to 100.
    :type epochs: int, optional
    :param target_polarizability: The specific polarizability value to target for generation.
                                  If None, a range of values will be targeted.
    :type target_polarizability: float, optional
    :param polymer_type: The type of molecule ('monomer' or 'copolymer').
                         Currently, only 'monomer' is supported. Defaults to 'monomer'.
    :type polymer_type: str, optional
    :param monomers_number_per_target: The number of molecules to generate for each target polarizability.
    :type monomers_number_per_target: int, optional
    :param threshold: The acceptable error margin for generated polarizability values.
    :type threshold: float, optional
    :return: None
    :rtype: None
    """
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
        targets = np.linspace(0, 1, 200)
    else:
        targets = [target_polarizability]
        
    prep = GenerativePreprocess(input_csv, generative_data_path)
    prep.run()
    trainer = GenerativeTrainer(generative_data_path, model_path, batch_size, learning_rate, epochs)
    trainer.run()
    generator = MoleculeGenerator(model_path, output_path, monomers_number_per_target, threshold)
    generator.run(targets)

def run_generative_ga_pipeline(input_csv='polygraphpy/data/polarizability_data.csv', prediction_target=None, polymer_type='monomer',
                               target_polarizability=None, population_size=100, generations=50, target_column='static_polarizability'):
    """Run the Genetic Algorithm (GA) based generative pipeline for molecule design.

    This function utilizes a genetic algorithm to evolve new molecules that have a
    specific target property, using a pre-trained GNN model for fitness evaluation.

    :param input_csv: Path to the input CSV file with SMILES and properties.
    :type input_csv: str, optional
    :param prediction_target: The column name of the property to target during generation.
    :type prediction_target: str
    :param polymer_type: The type of molecule ('monomer' or 'copolymer').
                         Currently, only 'monomer' is supported. Defaults to 'monomer'.
    :type polymer_type: str, optional
    :param target_polarizability: The specific polarizability value to aim for.
                                  If None, a range of values will be targeted.
    :type target_polarizability: float, optional
    :param population_size: The number of molecules in each GA generation. Defaults to 100.
    :type population_size: int, optional
    :param generations: The number of generations to run the GA for. Defaults to 50.
    :type generations: int, optional
    :return: None
    :rtype: None
    """
    if polymer_type != 'monomer':
        print("GA generative model currently supports only monomer.")
        return
    
    print('Starting generative model using GA...')
    
    train_input_data_path = 'polygraphpy/data/training_input_data/'
    gnn_output_path = 'polygraphpy/data/gnn_output/'

    print('Loading GNN model...')
    loader = GaModelLoader(input_csv, gnn_output_path, train_input_data_path, polymer_type, prediction_target)
    model, preprocess, atom_encoder, bond_encoder = loader.get_components()

    df = pd.read_csv(input_csv)
    df = df[df['chain_size'] == 0]

    scaler = MinMaxScaler()
    df['target_scaled'] = scaler.fit_transform(df[target_column].values.reshape(-1,1))

    if target_polarizability is None:
        Q1 = df['target_scaled'].quantile(0.25)
        Q3 = df['target_scaled'].quantile(0.75)
        targets = np.linspace(Q1, Q3, 100)
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

    df = pd.DataFrame(data)
    print('Saving generated molecules...')
    df.to_csv(os.path.join(output_path, 'generated_molecules.csv'), index=False)