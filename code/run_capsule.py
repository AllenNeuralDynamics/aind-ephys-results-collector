import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import shutil
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd

# SpikeInterface
import spikeinterface as si

# AIND
from aind_data_schema.core.data_description import (
    DataDescription,
    DerivedDataDescription,
    Institution,
    Modality,
    Modality,
    Platform,
    Funding,
    DataLevel,
)
from aind_data_schema.core.processing import DataProcess, Processing, PipelineProcess
from aind_data_schema.schema_upgrade.data_description_upgrade import DataDescriptionUpgrade
from aind_data_schema.schema_upgrade.processing_upgrade import ProcessingUpgrade, DataProcessUpgrade


PIPELINE_MAINAINER = "Alessio Buccino"
PIPELINE_URL = "https://github.com/AllenNeuralDynamics/aind-ephys-pipeline-pykilosort"
PIPELINE_VERSION = "0.1.0"


data_folder = Path("../data/")
results_folder = Path("../results/")

if __name__ == "__main__":
    ###### VISUALIZATION #########
    print("\n\nCOLLECTING RESULTS")
    t_collection_start = time.perf_counter()

    # check if test
    if (data_folder / "postprocessing_pipeline_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        postprocessed_folder = data_folder / "postprocessing_pipeline_output_test"
        preprocessed_folder = data_folder / "preprocessing_pipeline_output_test"
        spikesorted_folder = data_folder / "spikesorting_pipeline_output_test"
        curated_folder = data_folder / "curation_pipeline_output_test"
        unit_classifier_folder = data_folder / "unit_classifier_pipeline_output_test"
        visualization_folder = data_folder / "visualization_pipeline_output_test"

        test_folders = [
            preprocessed_folder,
            spikesorted_folder,
            postprocessed_folder,
            curated_folder,
            visualization_folder,
        ]

        data_processes_files = []
        for test_folder_name in test_folders:
            test_folder = data_folder / test_folder_name
            data_processes_files.extend(
                [p for p in test_folder.iterdir() if "data_process" in p.name and p.name.endswith(".json")]
            )
    else:
        postprocessed_folder = data_folder
        preprocessed_folder = data_folder
        spikesorted_folder = data_folder
        curated_folder = data_folder
        unit_classifier_folder = data_folder
        visualization_folder = data_folder
        data_processes_files = [
            p for p in data_folder.iterdir() if "data_process" in p.name and p.name.endswith(".json")
        ]

    ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    session = ecephys_sessions[0]
    session_name = session.name

    # Move spikesorted / postprocessing / curated
    spikesorted_results_folder = results_folder / "spikesorted"
    spikesorted_results_folder.mkdir(exist_ok=True)
    postprocessed_results_folder = results_folder / "postprocessed"
    postprocessed_results_folder.mkdir(exist_ok=True)
    curated_results_folder = results_folder / "curated"
    curated_results_folder.mkdir(exist_ok=True)

    spikesorted_folders = [p for p in spikesorted_folder.iterdir() if "spikesorted_" in p.name and p.is_dir()]
    for f in spikesorted_folders:
        shutil.copytree(f, spikesorted_results_folder / f.name[len("spikesorted_") :])
    postprocessed_folders = [
        p
        for p in postprocessed_folder.iterdir()
        if "postprocessed" in p.name and p.is_dir() and "-sorting" not in p.name
    ]
    for f in postprocessed_folders:
        recording_name = f.name[len("postprocessed_") :]
        shutil.copytree(f, postprocessed_results_folder / recording_name)

        # TODO: get sorting, add curation and unit_classifier properties and save to curated
        we = si.load_waveforms(f, with_recording=False)
        # add defaut_qc property
        curation_file = curated_folder / f"qc_{recording_name}.npy"
        if curation_file.is_file():
            default_qc = np.load(curation_file)
            we.sorting.set_property("default_qc", default_qc)
        # add classifier
        unit_classifier_file = unit_classifier_folder / f"unit_classifier_{recording_name}.csv"
        if unit_classifier_file.is_file():
            unit_classifier_df = pd.read_csv(unit_classifier_file, index_col=False)
            decoder_label = unit_classifier_df["decoder_label"]
            we.sorting.set_property("decoder_label", decoder_label)
            decoder_probability = unit_classifier_df["decoder_probability"]
            we.sorting.set_property("decoder_probability", decoder_probability)

        _ = we.sorting.save(folder=curated_results_folder / recording_name)

    postprocessed_sorting_folders = [
        p for p in postprocessed_folder.iterdir() if "postprocessed-sorting" in p.name and p.is_dir()
    ]
    for f in postprocessed_sorting_folders:
        shutil.copytree(f, postprocessed_results_folder / f.name)

    # Copy JSON preprocessed files
    preprocessed_json_files = [
        p for p in preprocessed_folder.iterdir() if "preprocessed_" in p.name and p.name.endswith(".json")
    ]
    (results_folder / "preprocessed").mkdir(exist_ok=True)
    for preprocessed_file in preprocessed_json_files:
        shutil.copy(preprocessed_file, results_folder / "preprocessed" / preprocessed_file.name[len("preprocessed_") :])

    # Make visualization_output
    visualization_output = {}
    visualization_json_files = [
        p
        for p in visualization_folder.iterdir()
        if "visualization_" in p.name and p.name.endswith(".json") and "data_process" not in p.name
    ]
    for visualization_json_file in visualization_json_files:
        recording_name = visualization_json_file.name[len("visualization_") : len(visualization_json_file.name) - 5]
        with open(visualization_json_file, "r") as f:
            visualization_dict = json.load(f)
        visualization_output[recording_name] = visualization_dict
    with open(results_folder / "visualization_output.json", "w") as f:
        json.dump(visualization_output, f, indent=4)

    # Collect and aggregate data processes and make Processing model
    ephys_data_processes = []
    for json_file in data_processes_files:
        with open(json_file, "r") as data_process_file:
            data_process_dict = json.load(data_process_file)
        data_process_old = DataProcess.model_construct(**data_process_dict)
        data_process = DataProcessUpgrade(data_process_old).upgrade()
        ephys_data_processes.append(data_process)

    if (session / "processing.json").is_file():
        with open(session / "processing.json", "r") as processing_file:
            processing_dict = json.load(processing_file)
        # Allow for parsing earlier versions of Processing files
        processing_old = Processing.model_construct(**processing_dict)
        # Protect against processing_pipeline.data_processes.outputs being None
        if hasattr(processing_old, "processing_pipeline"):
            processing_pipeline = processing_old.processing_pipeline
            if "data_processes" in processing_pipeline:
                data_processes = processing_pipeline["data_processes"]
                for data_process in data_processes:
                    if data_process["outputs"] is None:
                        data_process["outputs"] = dict()
        processing = ProcessingUpgrade(processing_old).upgrade(processor_full_name=PIPELINE_MAINAINER)
        processing.processing_pipeline.data_processes.append(ephys_data_processes)
    else:
        processing_pipeline = PipelineProcess(
            data_processes=ephys_data_processes,
            processor_full_name=PIPELINE_MAINAINER,
            pipeline_url=PIPELINE_URL,
            pipeline_version=PIPELINE_VERSION,
        )
        processing = Processing(processing_pipeline=processing_pipeline)

    with (results_folder / "processing.json").open("w") as f:
        f.write(processing.model_dump_json(indent=3))

    # Handle DataDescription model
    if (session / "data_description.json").is_file():
        with open(session / "data_description.json", "r") as data_description_file:
            data_description_json = json.load(data_description_file)
        # Allow for parsing earlier versions of Processing files
        data_description = DataDescription.model_construct(**data_description_json)
    else:
        data_description = None

    if (session / "subject.json").is_file():
        with open(session / "subject.json", "r") as subject_file:
            subject_info = json.load(subject_file)
        subject_id = subject_info["subject_id"]
    elif len(session_name.split("_")) > 1:
        subject_id = session_name.split("_")[1]
    else:
        subject_id = "000000"  # unknown

    process_name = "sorted"
    if data_description is not None:
        upgrader = DataDescriptionUpgrade(old_data_description_model=data_description)
        additional_required_kwargs = dict()
        # at least one investigator is required
        if len(data_description.investigators) == 0:
            additional_required_kwargs.update(dict(investigators=["Unknown"]))
        upgraded_data_description = upgrader.upgrade(platform=Platform.ECEPHYS, **additional_required_kwargs)
        derived_data_description = DerivedDataDescription.from_data_description(
            upgraded_data_description, process_name=process_name
        )
    else:
        # make from scratch:
        data_description_dict = {}
        data_description_dict["creation_time"] = datetime.now()
        data_description_dict["name"] = session_name
        data_description_dict["institution"] = Institution.AIND
        data_description_dict["data_level"] = DataLevel.RAW
        data_description_dict["investigators"] = ["Unknown"]
        data_description_dict["funding_source"] = [Funding(funder=Institution.AIND)]
        data_description_dict["modality"] = [Modality.ECEPHYS]
        data_description_dict["platform"] = Platform.ECEPHYS
        data_description_dict["subject_id"] = subject_id
        data_description = DataDescription(**data_description_dict)

        derived_data_description = DerivedDataDescription.from_data_description(
            data_description=data_description, process_name=process_name
        )

    # save processing files to output
    with (results_folder / "data_description.json").open("w") as f:
        f.write(derived_data_description.model_dump_json(indent=3))

    # Propagate other metadata JSON files
    metadata_json_files = [
        p
        for p in session.iterdir()
        if p.suffix == ".json" and "processing" not in p.name and "data_description" not in p.name
    ]
    for json_file in metadata_json_files:
        shutil.copy(json_file, results_folder)

    t_collection_end = time.perf_counter()
    elapsed_time_collection = np.round(t_collection_end - t_collection_start, 2)
    print(f"COLLECTION time: {elapsed_time_collection}s")
