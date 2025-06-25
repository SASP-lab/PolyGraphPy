from polygraphpy.pipelines import run_dftb_pipeline

if __name__ == "__main__":
    # Example usage of the DFTB+ pipeline with bundled example data
    results = run_dftb_pipeline(
        is_polymer=False,
        dftbplus_path=None,  # Optional: specify path to DFTB+ executable
        use_example_data=True  # Use bundled pubchem_dataset.csv
    )
    print("Pipeline completed:", results)
    
    # Example with custom input CSV
    # results = run_dftb_pipeline(
    #     input_csv='path/to/custom_dataset.csv',
    #     is_polymer=True,
    #     dftbplus_path='/path/to/dftb+'
    # )