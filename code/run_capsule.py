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
from aind_data_schema import Processing
from aind_data_schema.processing import DataProcess


PIPELINE_URL = "TBD"
PIPELINE_VERSION = "0.0.1"


data_folder = Path("../data/")
results_folder = Path("../results/")

if __name__ == "__main__":

    ###### VISUALIZATION #########
    print("\n\nCOLLECTING RESULTS")
    t_collection_start = time.perf_counter()

    # check if test
    if (data_folder / "postprocessing_output_test").is_dir():
        print("\n*******************\n**** TEST MODE ****\n*******************\n")
        test_folders = ["preprocessing_output_test", "spikesorting_output_test", "postprocessing_output_test",
                        "curation_output_test", "visualization_output_test"]
        postprocessed_folder = data_folder / "postprocessing_output_test" / "postprocessed"
        preprocessed_folder = data_folder / "preprocessing_output_test" / "preprocessed"
        spikesorted_folder = data_folder / "spikesorting_output_test" / "spikesorted"
        curated_folder = data_folder / "curation_output_test" / "sorting_precurated"
        visualization_folder = data_folder / "visualization_output_test" / "visualization_output"

        data_processes_folders = []
        for test_folder_name in test_folders:
            test_folder = data_folder / test_folder_name
            data_processes_folders.extend([p for p in test_folder.iterdir() if "data_processes" in p.name and p.is_dir()]) 
    else:
        postprocessed_folder = data_folder / "postprocessed"
        preprocessed_folder = data_folder / "preprocessed"
        spikesorted_folder = data_folder / "spikesorted"
        curated_folder = data_folder / "sorting_precurated"
        visualization_folder = data_folder / "visualization_output"
        data_processes_folders = [p for p in data_folder.iterdir() if "data_processes" in p.name and p.is_dir()]

    ecephys_sessions = [p for p in data_folder.iterdir() if "ecephys" in p.name.lower()]
    assert len(ecephys_sessions) == 1, f"Attach one session at a time {ecephys_sessions}"
    session = ecephys_sessions[0]
    session_name = session.name

    # Move spikesorted / postprocessing / curated
    shutil.copytree(spikesorted_folder, results_folder / "spikesorted")
    shutil.copytree(postprocessed_folder, results_folder / "postprocessed")
    shutil.copytree(curated_folder, results_folder / "sorting_precurated")

    # Copy JSON preprocessed files
    preprocessed_json_files = [p for p in preprocessed_folder.iterdir() if p.suffix == ".json"]
    (results_folder / "preprocessed").mkdir(exist_ok=True)
    for preprocessed_file in preprocessed_json_files:
        shutil.copy(preprocessed_file, results_folder / "preprocessed" / preprocessed_file.name)

    # Make visualization_output
    visualization_output = {}
    visualization_json_files = [p for p in visualization_folder.iterdir() if p.name.endswith(".json")]
    for visualization_json_file in visualization_json_files:
        recording_name = visualization_json_file.name[:visualization_json_file.name.find(".json")]
        with open(visualization_json_file, "r") as f:
            visualization_dict = json.load(f)
        visualization_output[recording_name] = visualization_dict
    with open(results_folder / "visualization_output.json", "w") as f:
        json.dump(visualization_output, f, indent=4)


    # Collect and aggregate data processes
    ephys_data_processes = []
    for data_processes_folder in data_processes_folders:
        json_files = [p for p in data_processes_folder.iterdir() if p.suffix == ".json"]
        for json_file in json_files:
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

    # Propagate other metadata
    metadata_json_files = [p for p in session.iterdir() if p.suffix == ".json" and "processing" not in p.name]
    for json_file in metadata_json_files:
        shutil.copy(json_file, results_folder)

    t_collection_end = time.perf_counter()
    elapsed_time_collection = np.round(t_collection_end - t_collection_start, 2)
    print(f"COLLECTION time: {elapsed_time_collection}s")
