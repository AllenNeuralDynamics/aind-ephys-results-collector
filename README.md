# Collect results for AIND ephys pipeline
## aind-ephys-results-collector


### Description

This simple capsule is designed to collect the results for the AIND pipeline. 


### Inputs

The `data/` folder can include several outputs from upstream capsules, including:

- outputs of the [aind-ephys-preprocessing](https://github.com/AllenNeuralDynamics/aind-ephys-preprocessing) capsule
- outputs of the spike sorting capsule (either with [aind-ephys-spikesort-pykilosort](https://github.com/AllenNeuralDynamics/aind-ephys-spikesort-pykilosort) or [aind-ephys-spikesort-kilosort25](https://github.com/AllenNeuralDynamics/aind-ephys-spikesort-kilosort25))
- outputs of the [aind-ephys-postprocessing](https://github.com/AllenNeuralDynamics/aind-ephys-postprocessing) capsule
- outputs of the [aind-ephys-curation](https://github.com/AllenNeuralDynamics/aind-ephys-curation) capsule
- outputs of the [aind-ephys-visualization](https://github.com/AllenNeuralDynamics/aind-ephys-visualization) capsule

In addition, the original session data (e.g., "ecephys_664438_2023-04-12_14-59-51") must also be available in the `data/` folder.



### Parameters

The `code/run` script takes no arguments.


### Output

The output of this capsule in the `results/` folder contains the following folders:

- `preprocessed`: collected JSON files from preprocessing
- `spikesorted`: collected sorted folders from spike sorting
- `postprocessed`: collected postprocessed folders from postprocessing
- `curated`: collected curated folders from curation

In addition, the following JSON files are produced:

- `visualization_output.json`: collected visualization links from visualization
- `data_description.json`: `DerivedDataDescription` object from [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package
- `processing.json`: `Processing` object from [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package
- other JSON files, propagated from input data (including, `subject.json`, `session.json`, etc.)
