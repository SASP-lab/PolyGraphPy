import click
from polygraphpy.pipelines import run_dftb_pipeline, run_gnn_pipeline

@click.command()

#DFTB parameters
@click.option('--input-csv', type=click.Path(exists=True), help='Path to input CSV file')
@click.option('--polymer', is_flag=True, help='Generate polymers instead of monomers')
@click.option('--dftbplus-path', default=None, type=click.Path(), help='Path to DFTB+ executable')
@click.option('--use-example-data', is_flag=True, help='Use bundled example data (reduced_dataset.csv)')
@click.option('--polymer-chain-size', default=2, type=int, help='Set polymer chain size (2, 3, 4) for polymer generation.')

#GNN parameters
@click.option('--batch-size',  default=16, type=int, help='Training batch size.')
@click.option('--learning-rate',  default=1e-3, type=float, help='Training learning rate.')
@click.option('--number-conv-channels',  default=225, type=int, help='Number of hidden channels in the convolutional layers.')
@click.option('--number-fc-channels',  default=225, type=int, help='Number of hidden channels in the MLP layer.')
@click.option('--prediction-target', help='Name of the target column from input data file.', required=True)
@click.option('--polymer-type', default='monomer', help='Type of polymers in the input data.')
@click.option("--epochs", default=200, type=int, help="Number of epochs to train the model.")

def main(input_csv, polymer, dftbplus_path, use_example_data, polymer_chain_size,
         batch_size, learning_rate, number_conv_channels, number_fc_channels, 
         prediction_target, polymer_type, epochs):
    
    """Run the PolyGraphPy DFTB+ and GNN pipelines for monomer or polymer simulations."""

    if input_csv and use_example_data:
        raise click.UsageError("Cannot use --input-csv with --use-example-data")
    
    # run_dftb_pipeline(
    #     input_csv=input_csv,
    #     is_polymer=polymer,
    #     dftbplus_path=dftbplus_path,
    #     use_example_data=use_example_data,
    #     polymer_chain_size=polymer_chain_size
    # )

    run_gnn_pipeline(
        input_csv=input_csv.split('/')[0] + '/' + input_csv.split('/')[1] + '/polarizability_data.csv',
        batch_size=batch_size,
        learning_rate=learning_rate,
        number_conv_channels=number_conv_channels,
        number_fc_channels=number_fc_channels,
        prediction_target=prediction_target,
        polymer_type=polymer_type,
        epochs=epochs
    )

if __name__ == '__main__':
    main()