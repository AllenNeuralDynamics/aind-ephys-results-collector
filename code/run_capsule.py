import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import shutil
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import os

# SpikeInterface
import spikeinterface as si

# AIND
from aind_data_schema.core.data_description import (
    DataDescription,
    DerivedDataDescription,
    Organization,
    Modality,
    Modality,
    Platform,
    Funding,
    DataLevel,
)
from aind_data_schema_models.pid_names import PIDName
from aind_data_schema.core.processing import DataProcess, Processing, PipelineProcess
from aind_metadata_upgrader.data_description_upgrade import DataDescriptionUpgrade
from aind_metadata_upgrader.processing_upgrade import ProcessingUpgrade, DataProcessUpgrade


PIPELINE_MAINAINER = "Alessio Buccino"
PIPELINE_URL = os.getenv("PIPELINE_URL")
PIPELINE_VERSION = os.getenv("PIPELINE_VERSION")


data_folder = Path("../data/")
results_folder = Path("../results/")

if __name__ == "__main__":
    ###### COLLECT RESULTS #########
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

    json_files = [p for p in data_folder.iterdir() if "job" in p.name and p.suffix == ".json"]
    # load session_name from any JSON file
    if len(json_files) > 0:
        with open(json_files[0], "r") as f:
            job_dict = json.load(f)
        session_name = job_dict["session_name"]
        print(f"Loaded session name from JSON files: {session_name}")
    else:
        session_name = session.name

    # Move spikesorted / postprocessing / curated
    spikesorted_results_folder = results_folder / "spikesorted"
    spikesorted_results_folder.mkdir(exist_ok=True)
    preprocessed_results_folder = results_folder / "preprocessed"
    preprocessed_results_folder.mkdir(exist_ok=True)
    postprocessed_results_folder = results_folder / "postprocessed"
    postprocessed_results_folder.mkdir(exist_ok=True)
    curated_results_folder = results_folder / "curated"
    curated_results_folder.mkdir(exist_ok=True)

    # PREPROCESSED
    print("Copying preprocessed folders to results:")
    preprocessed_json_files = [
        p for p in preprocessed_folder.iterdir() if "preprocessed_" in p.name and p.name.endswith(".json")
    ]
    (results_folder / "preprocessed").mkdir(exist_ok=True)
    for preprocessed_file in preprocessed_json_files:
        recording_json_file_name = preprocessed_file.name[len("preprocessed_") :]
        recording_name = preprocessed_file.stem[len("preprocessed_") :]
        recording_output_json_file = preprocessed_results_folder / recording_json_file_name
        print(f"\t{recording_name}")
        if session_name == "ecephys_session":
            shutil.copy(preprocessed_file, recording_output_json_file)
        else:
            print(f"\tRemapping preprocessed recording JSON path")
            with open(preprocessed_file, "r") as f:
                recording_dict = json.load(f)
            recording_dict_str = json.dumps(recording_dict, indent=4).replace("ecephys_session", session_name)
            recording_output_json_file.write_text(recording_dict_str, encoding="utf8")

    # MOTION
    motion_folders = [
        p for p in preprocessed_folder.iterdir() if "motion_" in p.name and p.is_dir()
    ]
    if len(motion_folders) > 0:
        print("Copying motion folders to results:")
        motion_results_folder = preprocessed_results_folder / "motion"
        motion_results_folder.mkdir(exist_ok=True)
        for motion_folder in motion_folders:
            recording_name = motion_folder.name[len("motion_") :]
            print(f"\t{recording_name}")
            shutil.copytree(motion_folder, motion_results_folder / recording_name)

    # SPIKESORTED
    print("Copying spikesorted folders to results:")
    spikesorted_folders = [p for p in spikesorted_folder.iterdir() if "spikesorted_" in p.name and p.is_dir()]
    for f in spikesorted_folders:
        recording_name = f.name[len("spikesorted_") :]
        print(f"\t{recording_name}")
        shutil.copytree(f, spikesorted_results_folder / recording_name)

    # POSTPROCESSED / CURATED
    print("Copying postprocessed and curated folders to results:")
    postprocessed_folders = [
        p
        for p in postprocessed_folder.iterdir()
        if "postprocessed" in p.name and p.is_dir()
    ]
    for f in postprocessed_folders:
        recording_name = f.stem[len("postprocessed_") :]
        analyzer_output_folder = None
        print(f"\t{recording_name}")
        try:
            analyzer = si.load_sorting_analyzer(f)
            if f.name.endswith(".zarr"):
                recording_folder_name = f"{recording_name}.zarr"
                analyzer_format = "zarr"
            else:
                recording_folder_name = recording_name
                analyzer_format = "binary_folder"
            shutil.copytree(f, postprocessed_results_folder / recording_folder_name)
            analyzer_output_folder = postprocessed_results_folder / recording_folder_name 
        except:
            print(f"Spike sorting failed on {recording_name}. Skipping collection")
            continue
        
        # add defaut_qc property
        default_qc = None
        decoder_label = None
        curation_file = curated_folder / f"qc_{recording_name}.npy"
        if curation_file.is_file():
            default_qc = np.load(curation_file)
            analyzer.sorting.set_property("default_qc", default_qc)
        # add classifier
        unit_classifier_file = unit_classifier_folder / f"unit_classifier_{recording_name}.csv"
        if unit_classifier_file.is_file():
            unit_classifier_df = pd.read_csv(unit_classifier_file, index_col=False)
            decoder_label = np.array(unit_classifier_df["decoder_label"].values).astype("str")
            analyzer.sorting.set_property("decoder_label", decoder_label)
            decoder_probability = np.array(unit_classifier_df["decoder_probability"].values).astype(float)
            analyzer.sorting.set_property("decoder_probability", decoder_probability)

        _ = analyzer.sorting.save(folder=curated_results_folder / recording_name)

        # update analyzer properties
        if analyzer_format == "binary_folder":
            if default_qc is not None or decoder_label is not None:
                _ = analyzer.sorting.save(folder=analyzer_output_folder / "sorting", overwrite=True)
            # update recording JSON path
            recording_json_path = analyzer_output_folder / "recording.json"
            if recording_json_path.is_file() and session_name != "ecephys_session":
                with open(recording_json_path, "r") as f:
                    recording_dict = json.load(f)
                recording_dict_str = json.dumps(recording_dict, indent=4)
                if "ecephys_session" in recording_dict_str:
                    print(f"\tRemapping analyzer recording path")
                    recording_dict_str = recording_dict_str.replace("ecephys_session", session_name)
                    recording_json_path.write_text(recording_dict_str, encoding="utf8")
        else:
            import zarr
            import numcodecs

            analyzer_root = zarr.open(analyzer_output_folder, mode="r+")

            if default_qc is not None or decoder_label is not None:
                from spikeinterface.core.zarrextractors import add_sorting_to_zarr_group
                del analyzer_root["sorting"]
                add_sorting_to_zarr_group(analyzer.sorting, analyzer_root.create_group("sorting"))

            # update recording field if is JSON
            if session_name != "ecephys_session":
                recording_root = analyzer_root["recording"]
                object_codec = None
                if isinstance(recording_root.filters[0], numcodecs.JSON):
                    object_codec = numcodecs.JSON()
                elif isinstance(recording_root.filters[0], numcodecs.Pickle):
                    object_codec = numcodecs.Pickle()
                if object_codec is not None:
                    recording_dict = recording_root[0]
                    recording_dict_str = json.dumps(recording_dict, indent=4)
                    if "ecephys_session" in recording_dict_str:
                        print(f"\tRemapping analyzer recording path")
                        recording_dict_mapped = json.loads(
                            recording_dict_str.replace("ecephys_session", session_name)
                        )
                        del analyzer_root["recording"]
                        zarr_rec = np.array([recording_dict_mapped], dtype=object)
                        analyzer_root.create_dataset("recording", data=zarr_rec, object_codec=object_codec)
                else:
                    print(f"Unsupported recording object codec: {recording_root.filters[0]}. Cannot remap recording path")

    postprocessed_sorting_folders = [
        p for p in postprocessed_folder.iterdir() if "postprocessed-sorting" in p.name and p.is_dir()
    ]
    for f in postprocessed_sorting_folders:
        shutil.copytree(f, postprocessed_results_folder / f.name)


    # VISUALIZATION
    print("Copying visualization outputs to results:")
    visualization_output = {}
    visualization_json_files = [
        p
        for p in visualization_folder.iterdir()
        if "visualization_" in p.name and p.name.endswith(".json") and "data_process" not in p.name
    ]
    for visualization_json_file in visualization_json_files:
        recording_name = visualization_json_file.name[len("visualization_") : len(visualization_json_file.name) - 5]
        print(f"\t{recording_name}")
        with open(visualization_json_file, "r") as f:
            visualization_dict = json.load(f)
        visualization_output[recording_name] = visualization_dict
    with open(results_folder / "visualization_output.json", "w") as f:
        json.dump(visualization_output, f, indent=4)

    # PROCESSING
    print("Generating processing metadata")
    ephys_data_processes = []
    for json_file in data_processes_files:
        with open(json_file, "r") as data_process_file:
            data_process_dict = json.load(data_process_file)
        data_process_old = DataProcess.model_construct(**data_process_dict)
        data_process = DataProcessUpgrade(data_process_old).upgrade()
        ephys_data_processes.append(data_process)

    processing = None
    if (session / "processing.json").is_file():
        with open(session / "processing.json", "r") as processing_file:
            processing_dict = json.load(processing_file)
        try:
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
            processing.processing_pipeline.pipeline_url = PIPELINE_URL
            processing.processing_pipeline.pipeline_version = PIPELINE_VERSION
        except Exception as e:
            print(f"Failed upgrading processing for error:\n{e}\nCreating from scratch.")
            processing = None

    if processing is None:
        processing_pipeline = PipelineProcess(
            data_processes=ephys_data_processes,
            processor_full_name=PIPELINE_MAINAINER,
            pipeline_url=PIPELINE_URL,
            pipeline_version=PIPELINE_VERSION,
        )
        processing = Processing(processing_pipeline=processing_pipeline)

    with (results_folder / "processing.json").open("w") as f:
        f.write(processing.model_dump_json(indent=3))

    # DATA_DESCRIPTION
    print("Generating data_description metadata")
    data_description = None
    if (session / "data_description.json").is_file():
        with open(session / "data_description.json", "r") as data_description_file:
            data_description_json = json.load(data_description_file)
        # Allow for parsing earlier versions of Processing files
        data_description = DataDescription.model_construct(**data_description_json)

    if (session / "subject.json").is_file():
        with open(session / "subject.json", "r") as subject_file:
            subject_info = json.load(subject_file)
        subject_id = subject_info["subject_id"]
    else:
        subject_id = "000000"  # unknown

    process_name = "sorted"
    if data_description is not None:
        try:
            upgrader = DataDescriptionUpgrade(data_description)
            additional_required_kwargs = dict()
            # at least one investigator is required
            if len(data_description.investigators) == 0:
                additional_required_kwargs.update(dict(investigators=["Unknown"]))
            if len(data_description.funding_source) == 0:
                additional_required_kwargs.update(
                    dict(funding_source=[Funding(funder=Organization.AI)])
                )
            upgraded_data_description = upgrader.upgrade(platform=Platform.ECEPHYS, **additional_required_kwargs)
            derived_data_description = DerivedDataDescription.from_data_description(
                upgraded_data_description, process_name=process_name
            )
        except Exception as e:
            print(f"Failed upgrading data description for error:\n{e}\nCreating from scratch.")
            data_description = None
    if data_description is None:
        # make from scratch:
        data_description_dict = {}
        data_description_dict["creation_time"] = datetime.now()
        data_description_dict["name"] = session_name
        data_description_dict["institution"] = Organization.AIND
        data_description_dict["data_level"] = DataLevel.RAW
        data_description_dict["investigators"] = [PIDName(name="Unknown")]
        data_description_dict["funding_source"] = [Funding(funder=Organization.AI)]
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

    # OTHER METADATA FILES
    print("Propagating other metadata files")
    metadata_json_files = [
        p
        for p in session.iterdir()
        if p.suffix == ".json" and "processing" not in p.name and "data_description" not in p.name and "job" not in p.name
    ]
    for json_file in metadata_json_files:
        shutil.copy(json_file, results_folder)

    t_collection_end = time.perf_counter()
    elapsed_time_collection = np.round(t_collection_end - t_collection_start, 2)
    print(f"COLLECTION time: {elapsed_time_collection}s")
