import click
from polygraphpy.pipelines import run_dftb_pipeline, run_gnn_pipeline, run_generative_pipeline, run_generative_ga_pipeline

@click.command()

#DFTB parameters
@click.option('--run-dftb', is_flag=True, help='Run the DFTB+ to make simulate monomers and polymers.')
@click.option('--input-csv', type=click.Path(exists=True), default='polygraphpy/data/original_dataset.csv', help='Path to input CSV file.')
@click.option('--is-polymer', is_flag=True, help='Generate polymers instead of monomers.')
@click.option('--polymer-type', default='monomer', type=click.Choice(['monomer', 'homopolymer', 'copolymer']), 
                                help='Type of polymers in the input data.')
@click.option('--dftbplus-path', default=None, type=click.Path(), help='Path to DFTB+ executable.')
@click.option('--use-example-data', is_flag=True, help='Use bundled example data (reduced_dataset.csv)')
@click.option('--polymer-chain-size', default=2, type=int, help='Set polymer chain size (1, 2, 3, 4, ...) for polymer generation.')

#GNN parameters
@click.option('--train-gnn-prediction', is_flag=True, help='Train the GNN model to make property predictions.')
@click.option('--batch-size', default=32, type=int, help='Training batch size.')
@click.option('--learning-rate', default=5e-4, type=float, help='Training learning rate.')
@click.option('--number-conv-channels', default=150, type=int, help='Number of hidden channels in the convolutional layers.')
@click.option('--number-fc-channels', default=150, type=int, help='Number of hidden channels in the MLP layer.')
@click.option('--prediction-target', default='static_polarizability', help='Name of the target column from input data file.')
@click.option("--epochs", default=200, type=int, help="Number of epochs to train the model.")
@click.option("--prediction-model", default='gunet', type=click.Choice(['gcn', 'gunet']), help="Neural net model to train.")

#Generative parameters
@click.option('--run-generative', is_flag=True, help='Run the generative model pipeline.')
@click.option('--generative-model', default='gpt', type=click.Choice(['gpt', 'ga']), help='Generative model to use: GPT-based or Genetic Algorithm.')
@click.option('--target-polarizability', default=None, type=float, help='Target scaled polarizability (0-1). If not provided, use linspace(0,1,200).')
@click.option('--generative-batch-size', default=4, type=int, help='Batch size for GPT training.')
@click.option('--generative-learning-rate', default=5e-5, type=float, help='Learning rate for GPT training.')
@click.option('--generative-epochs', default=150, type=int, help='Number of epochs for GPT training.')
@click.option('--monomers-number-per-target', default=1, type=int, help='Number of monomers for GPT generation per target value.')
@click.option('--threshold', default=1e-1, type=int, help='Error threshold for GPT generation.')
@click.option('--ga-population-size', default=100, type=int, help='Population size for GA.')
@click.option('--ga-generations', default=50, type=int, help='Number of generations for GA.')

def main(run_dftb, input_csv, is_polymer, polymer_type, dftbplus_path, use_example_data, polymer_chain_size, train_gnn_prediction, batch_size, learning_rate, 
         number_conv_channels, number_fc_channels, prediction_target, epochs, run_generative, generative_model, target_polarizability, generative_batch_size,
         generative_learning_rate, generative_epochs, monomers_number_per_target, threshold, ga_population_size, ga_generations, prediction_model):
    
    """Run the PolyGraphPy DFTB+ and GNN pipelines for monomer or polymer simulations."""

    if use_example_data:
        input_csv = None

    if input_csv and use_example_data:
        raise click.UsageError("Cannot use --input-csv with --use-example-data")
    
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