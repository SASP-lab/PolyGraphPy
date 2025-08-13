"""
polygraphpy.__main__
====================

Command-line interface (CLI) for the PolyGraphPy package.

This module defines the `main()` entry point, which orchestrates the
package's pipelines for:

- **DFTB+ simulations** of monomers, homopolymers, and copolymers.
- **Graph Neural Network (GNN)** training and prediction of molecular properties.
- **Generative model pipelines** for molecular design using either a GPT-based model or
  a Genetic Algorithm (GA).

The CLI is implemented using the `click` library, and all options can be
viewed by running:

.. code-block:: console

    $ python -m polygraphpy --help

Example usage:

.. code-block:: console

    # Run DFTB+ simulation with example data
    python -m polygraphpy --run-dftb --use-example-data --is-polymer --polymer-type homopolymer

    # Train GNN model on polarizability data
    python -m polygraphpy --train-gnn-prediction --input-csv path/to/data.csv --prediction-target static_polarizability

    # Run GPT-based generative pipeline targeting a polarizability of 0.75
    python -m polygraphpy --run-generative --generative-model gpt --target-polarizability 0.75

"""

import click
from polygraphpy.pipelines import run_dftb_pipeline, run_gnn_pipeline, run_generative_pipeline, run_generative_ga_pipeline

@click.command()

# DFTB parameters
@click.option('--run-dftb', 
              is_flag=True, 
              help='Run the DFTB+ simulation pipeline to generate molecular geometries and compute properties.')

@click.option('--input-csv', 
              type=click.Path(exists=True), 
              default='polygraphpy/data/original_dataset.csv', 
              help='Path to the input CSV file containing SMILES strings. Required unless --use-example-data is set.')

@click.option('--is-polymer', 
              is_flag=True, 
              help='Flag to indicate that the input molecules are polymers, not monomers.')

@click.option('--polymer-type', 
              default='monomer', 
              type=click.Choice(['monomer', 'homopolymer', 'copolymer']), 
              help='Specifies the type of polymer to generate (monomer, homopolymer, or copolymer). Here we consider monomer as polymer with chain_size = 0.')

@click.option('--dftbplus-path', 
              default=None, 
              type=click.Path(), 
              help='Path to the DFTB+ executable. Must be provided if it is not in the system PATH.')

@click.option('--use-example-data', 
              is_flag=True, 
              help='Use a bundled, small example dataset (`reduced_dataset.csv`) for testing purposes.')

@click.option('--polymer-chain-size', 
              default=2, 
              type=int, 
              help='Sets the number of repeating monomer units for polymer generation.')

# GNN parameters
@click.option('--train-gnn-prediction', 
              is_flag=True, 
              help='Run the GNN pipeline to train a model for property prediction.')

@click.option('--batch-size', 
              default=32, 
              type=int, 
              help='The batch size to be used during GNN model training.')

@click.option('--learning-rate', 
              default=5e-4, 
              type=float, 
              help='The learning rate for the GNN model optimizer.')

@click.option('--number-conv-channels', 
              default=150, 
              type=int, 
              help='The number of hidden channels in the GCN convolutional layers.')

@click.option('--number-fc-channels', 
              default=150, 
              type=int, 
              help='The number of hidden channels in the GNNs fully connected (MLP) layers.')

@click.option('--prediction-target', 
              default='static_polarizability', 
              help='The name of the column in the input data to be used as the prediction target.')

@click.option("--epochs", 
              default=200, 
              type=int, 
              help="The number of epochs to train the GNN model.")

@click.option("--prediction-model", 
              default='gunet', 
              type=click.Choice(['gcn', 'gunet']), 
              help="Specifies the GNN model architecture to be trained ('gcn' or 'gunet').")

# Generative parameters
@click.option('--run-generative', 
              is_flag=True, 
              help='Run the generative model pipeline to design new molecules with desired properties.')

@click.option('--generative-model', 
              default='gpt', 
              type=click.Choice(['gpt', 'ga']), 
              help='Specifies which generative model to use: a GPT-based sequence model or a Genetic Algorithm (GA).')

@click.option('--target-polarizability', 
              default=None, 
              type=float, 
              help='A specific scaled polarizability value (between 0 and 1) to target for molecule generation. If not specified, a range of values is targeted.')

@click.option('--generative-batch-size', 
              default=4, 
              type=int, 
              help='The batch size for training the GPT generative model.')

@click.option('--generative-learning-rate', 
              default=5e-5, 
              type=float,
              help='The learning rate for the GPT generative model.')

@click.option('--generative-epochs', 
              default=150, 
              type=int, 
              help='The number of epochs to train the GPT generative model.')

@click.option('--monomers-number-per-target', 
              default=1, 
              type=int, 
              help='The number of monomers to generate for each target polarizability value (for the GPT model).')

@click.option('--threshold', 
              default=3e-1, 
              type=float, 
              help='The acceptable error threshold for the generated molecule property prediction (for the GPT model).')

@click.option('--ga-population-size', 
              default=100, 
              type=int, 
              help='The population size for the Genetic Algorithm.')

@click.option('--ga-generations', 
              default=50, 
              type=int, 
              help='The number of generations to run the Genetic Algorithm for.')

def main(run_dftb, 
         input_csv, 
         is_polymer, 
         polymer_type, 
         dftbplus_path, 
         use_example_data, 
         polymer_chain_size, 
         train_gnn_prediction, 
         batch_size, 
         learning_rate, 
         number_conv_channels, 
         number_fc_channels, 
         prediction_target, 
         epochs, 
         run_generative, 
         generative_model, 
         target_polarizability, 
         generative_batch_size,
         generative_learning_rate, 
         generative_epochs, 
         monomers_number_per_target, 
         threshold, 
         ga_population_size, 
         ga_generations, 
         prediction_model):
    
    """Main CLI entry point for the PolyGraphPy package.

    This function provides a command-line interface to execute the different
    pipelines of the PolyGraphPy package, including DFTB+ simulations, GNN
    property prediction, and generative molecular design.

    :param run_dftb: Flag to run the DFTB+ simulation pipeline.
    :type run_dftb: bool
    :param input_csv: Path to the input CSV file.
    :type input_csv: str
    :param is_polymer: Flag to indicate polymer generation.
    :type is_polymer: bool
    :param polymer_type: Type of polymer to simulate.
    :type polymer_type: str
    :param dftbplus_path: Path to the DFTB+ executable.
    :type dftbplus_path: str, optional
    :param use_example_data: Flag to use the bundled example data.
    :type use_example_data: bool
    :param polymer_chain_size: Number of units in the polymer chain.
    :type polymer_chain_size: int
    :param train_gnn_prediction: Flag to train the GNN prediction model.
    :type train_gnn_prediction: bool
    :param batch_size: Batch size for GNN training.
    :type batch_size: int
    :param learning_rate: Learning rate for GNN training.
    :type learning_rate: float
    :param number_conv_channels: Number of hidden channels in GNN conv layers.
    :type number_conv_channels: int
    :param number_fc_channels: Number of hidden channels in GNN FC layers.
    :type number_fc_channels: int
    :param prediction_target: Name of the target column for prediction.
    :type prediction_target: str
    :param epochs: Number of GNN training epochs.
    :type epochs: int
    :param run_generative: Flag to run the generative model pipeline.
    :type run_generative: bool
    :param generative_model: The generative model to use ('gpt' or 'ga').
    :type generative_model: str
    :param target_polarizability: The target polarizability for generation.
    :type target_polarizability: float, optional
    :param generative_batch_size: Batch size for GPT training.
    :type generative_batch_size: int
    :param generative_learning_rate: Learning rate for GPT training.
    :type generative_learning_rate: float
    :param generative_epochs: Number of epochs for GPT training.
    :type generative_epochs: int
    :param monomers_number_per_target: Number of monomers to generate per target value.
    :type monomers_number_per_target: int
    :param threshold: Error threshold for GPT generation.
    :type threshold: float
    :param ga_population_size: Population size for the Genetic Algorithm.
    :type ga_population_size: int
    :param ga_generations: Number of generations for the Genetic Algorithm.
    :type ga_generations: int
    :param prediction_model: The GNN model architecture to use.
    :type prediction_model: str
    :raises click.UsageError: If both `input_csv` and `use_example_data` are provided.
    """

    if input_csv and use_example_data:
        raise click.UsageError("Cannot use --input-csv with --use-example-data")

    if use_example_data:
        input_csv = None
    
    if run_dftb:
        run_dftb_pipeline(
            input_csv=input_csv,
            is_polymer=is_polymer,
            polymer_type=polymer_type,
            dftbplus_path=dftbplus_path,
            use_example_data=use_example_data,
            polymer_chain_size=polymer_chain_size
        )
    else:
        print('Jumping DFTB+ execution.')

    aux = ''
    if polymer_type == 'copolymer':
        aux = '_copoly'

    if train_gnn_prediction:
        if prediction_target is not None:
            run_gnn_pipeline(
                input_csv=input_csv.split('/')[0] + '/' + input_csv.split('/')[1] + f'/polarizability_data{aux}.csv',
                batch_size=batch_size,
                learning_rate=learning_rate,
                number_conv_channels=number_conv_channels,
                number_fc_channels=number_fc_channels,
                prediction_target=prediction_target,
                polymer_type=polymer_type,
                epochs=epochs,
                model=prediction_model
            )
        else:
            print('Make sure that you provide the target for train and prediction.')
    else:
        print('Jumping GNN training and prediction.')

    if run_generative:
        gen_input_csv = input_csv.split('/')[0] + '/' + input_csv.split('/')[1] + '/polarizability_data.csv'

        if generative_model == 'gpt':
            run_generative_pipeline(
                input_csv=gen_input_csv,
                batch_size=generative_batch_size,
                learning_rate=generative_learning_rate,
                epochs=generative_epochs,
                target_polarizability=target_polarizability,
                polymer_type=polymer_type,
                monomers_number_per_target=monomers_number_per_target,
                threshold=threshold,
            )

        else:
            run_generative_ga_pipeline(
                input_csv=gen_input_csv,
                prediction_target=prediction_target,
                polymer_type=polymer_type,
                target_polarizability=target_polarizability,
                population_size=ga_population_size,
                generations=ga_generations
            )

if __name__ == '__main__':
    main()