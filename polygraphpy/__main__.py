import click
from polygraphpy.pipelines import run_dftb_pipeline

@click.command()
@click.option('--input-csv', type=click.Path(exists=True), help='Path to input CSV file')
@click.option('--polymer', is_flag=True, help='Generate polymers instead of monomers')
@click.option('--dftbplus-path', default=None, type=click.Path(), help='Path to DFTB+ executable')
@click.option('--use-example-data', is_flag=True, help='Use bundled example data (pubchem_dataset.csv)')

def main(input_csv, polymer, dftbplus_path, use_example_data):
    """Run the PolyGraphPy DFTB+ pipeline for monomer or polymer simulations."""
    if input_csv and use_example_data:
        raise click.UsageError("Cannot use --input-csv with --use-example-data")
    run_dftb_pipeline(
        input_csv=input_csv,
        is_polymer=polymer,
        dftbplus_path=dftbplus_path,
        use_example_data=use_example_data
    )

if __name__ == '__main__':
    main()