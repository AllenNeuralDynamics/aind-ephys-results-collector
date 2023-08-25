import warnings
warnings.filterwarnings("ignore")


from pathlib import Path
import numpy as np
import shutil
import json
import sys
import time
from datetime import datetime, timedelta

# AIND
import aind_data_schema.data_description as dd
from aind_data_schema.processing import DataProcess, Processing
from aind_data_schema.schema_upgrade.data_description_upgrade import DataDescriptionUpgrade


PIPELINE_URL = "TBD"
PIPELINE_VERSION = "0.0.1"


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
        visualization_folder = data_folder / "visualization_pipeline_output_test"

        test_folders = [preprocessed_folder, spikesorted_folder, postprocessed_folder, curated_folder, visualization_folder]

        data_processes_files = []
        for test_folder_name in test_folders:
            test_folder = data_folder / test_folder_name
            data_processes_files.extend([p for p in test_folder.iterdir() if "data_process" in p.name and p.name.endswith(".json")]) 
    else:
        postprocessed_folder = data_folder
        preprocessed_folder = data_folder
        spikesorted_folder = data_folder
        curated_folder = data_folder
        visualization_folder = data_folder 
        data_processes_files = [p for p in data_folder.iterdir() if "data_process" in p.name and p.name.endswith(".json")]

    ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    session = ecephys_sessions[0]
    session_name = session.name

    # Move spikesorted / postprocessing / curated
    spikesorted_results_folder = (results_folder / "spikesorted")
    spikesorted_results_folder.mkdir(exist_ok=True)
    postprocessed_results_folder = (results_folder / "postprocessed")
    postprocessed_results_folder.mkdir(exist_ok=True)
    curated_results_folder = (results_folder / "curated")
    curated_results_folder.mkdir(exist_ok=True)

    spikesorted_folders = [p for p in spikesorted_folder.iterdir() if "spikesorted_" in p.name and p.is_dir()]
    for f in spikesorted_folders:
        shutil.copytree(f, spikesorted_results_folder / f.name[len("spikesorted_"):])
    postprocessed_folders = [p for p in postprocessed_folder.iterdir() if "postprocessed" in p.name and p.is_dir() and "-sorting" not in p.name]
    for f in postprocessed_folders:
        shutil.copytree(f, postprocessed_results_folder / f.name[len("postprocessed_"):])
    postprocessed_sorting_folders = [p for p in postprocessed_folder.iterdir() if "postprocessed-sorting" in p.name and p.is_dir()]
    for f in postprocessed_sorting_folders:
        shutil.copytree(f, postprocessed_results_folder / f.name)
    curated_folders = [p for p in curated_folder.iterdir() if "curated_" in p.name and p.is_dir()]
    for f in curated_folders:
        shutil.copytree(f, curated_results_folder / f.name[len("curated_"):])

    # Copy JSON preprocessed files
    preprocessed_json_files = [p for p in preprocessed_folder.iterdir() if "preprocessed_" in p.name and p.name.endswith(".json")]
    (results_folder / "preprocessed").mkdir(exist_ok=True)
    for preprocessed_file in preprocessed_json_files:
        shutil.copy(preprocessed_file, results_folder / "preprocessed" / preprocessed_file.name[len("preprocessed_"):])

    # Make visualization_output
    visualization_output = {}
    visualization_json_files = [p for p in visualization_folder.iterdir() if "visualization_" in p.name and p.name.endswith(".json") and "data_process" not in p.name]
    for visualization_json_file in visualization_json_files:
        recording_name = visualization_json_file.name[len("visualization_"):len(visualization_json_file.name) - 5]
        with open(visualization_json_file, "r") as f:
            visualization_dict = json.load(f)
        visualization_output[recording_name] = visualization_dict
    with open(results_folder / "visualization_output.json", "w") as f:
        json.dump(visualization_output, f, indent=4)


    # Collect and aggregate data processes
    ephys_data_processes = []
    for json_file in data_processes_files:
        data_process = DataProcess.parse_file(json_file)
        ephys_data_processes.append(data_process)

    # Make Processing
    ephys_processing = Processing(
            pipeline_url=PIPELINE_URL,
            pipeline_version=PIPELINE_VERSION,
            data_processes=ephys_data_processes
        )
    if (session / "processing.json").is_file():
        with open(session / "processing.json", "r") as processing_file:
            processing_json = json.load(processing_file)
        # Allow for parsing earlier versions of Processing files
        processing = Processing.construct(**processing_json)
    else:
        processing = None

    if processing is None:
        processing = ephys_processing
    else:
        processing.data_processes.append(ephys_data_processes)
    with (results_folder / "processing.json").open("w") as f:
        f.write(processing.json(indent=3))

    if (session / "data_description.json").is_file():
        with open(session / "data_description.json", "r") as data_description_file:
            data_description_json = json.load(data_description_file)
        # Allow for parsing earlier versions of Processing files
        data_description = DataDescription.construct(**data_description_json)
    else:
        data_description = None

    if (session / "subject.json").is_file():
        with open(session / "subject.json", "r") as subject_file:
            subject_info = json.load(subject_file)
        subject_id = subject_info["subject_id"]
    elif len(session_name.split("_")) > 1:
        subject_id = session_name.split("_")[1]
    else:
        subject_id = "000000" # unknown

    process_name = "sorted"
    if data_description is not None:
        upgrader = DataDescriptionUpgrade(old_data_description_model=data_description)
        upgraded_data_description = upgrader.upgrade_data_description(experiment_type=dd.ExperimentType.ECEPHYS)
        derived_data_description = dd.DerivedDataDescription.from_data_description(
            upgraded_data_description, process_name=process_name
        )
    else:
        now = datetime.now()
        # make from scratch:
        data_description_dict = {}
        data_description_dict["creation_time"] = now.time()
        data_description_dict["creation_date"] = now.date()
        data_description_dict["input_data_name"] = session_name
        data_description_dict["institution"] = dd.Institution.AIND
        data_description_dict["investigators"] = []
        data_description_dict["funding_source"] = [dd.Funding(funder="AIND")]
        data_description_dict["modality"] = [dd.Modality.ECEPHYS]
        data_description_dict["experiment_type"] = dd.ExperimentType.ECEPHYS
        data_description_dict["subject_id"] = subject_id

        derived_data_description = dd.DerivedDataDescription(process_name=process_name, **data_description_dict)

    # save processing files to output
    with (results_folder / "data_description.json").open("w") as f:
        f.write(derived_data_description.json(indent=3))

    # Propagate other metadata
    metadata_json_files = [p for p in session.iterdir() if p.suffix == ".json" and "processing" not in p.name and "data_description" not in p.name]
    for json_file in metadata_json_files:
        shutil.copy(json_file, results_folder)

    t_collection_end = time.perf_counter()
    elapsed_time_collection = np.round(t_collection_end - t_collection_start, 2)
    print(f"COLLECTION time: {elapsed_time_collection}s")
