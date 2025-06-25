from polygraphpy.pipelines import run_dftb_pipeline

if __name__ == "__main__":
    # Example usage of the DFTB+ pipeline for monomers
    results = run_dftb_pipeline(
        input_csv='pubchem_dataset.csv',
        is_polymer=False,
        dftbplus_path=None  # Optional: specify path to DFTB+ executable
    )
    print("Pipeline completed:", results)