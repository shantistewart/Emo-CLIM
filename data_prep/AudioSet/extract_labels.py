"""Script to extract labels for AudioSet music mood subset."""


import os
import pandas as pd
import json
import tqdm
import warnings


# data paths:
DATA_ROOT = "/proj/systewar/datasets/audioset_music_mood/audio_files"
ONTOLOGY_FILE = "/proj/systewar/datasets/audioset_music_mood/orig_metadata_files/ontology.json"
METADATA_FILES = {
    "train": "/proj/systewar/datasets/audioset_music_mood/orig_metadata_files/unbalanced_train_segments.csv",
    "eval": "/proj/systewar/datasets/audioset_music_mood/orig_metadata_files/eval_segments.csv"
}
# AudioSet music mood subset label names:
MOOD_LABEL_NAMES = ["Happy music", "Funny music", "Sad music", "Tender music", "Exciting music", "Angry music", "Scary music"]

# script options:
data_subsets = ["train", "eval"]
label_files = {
    "train": "/proj/systewar/datasets/audioset_music_mood/labels_train.csv",
    "eval": "/proj/systewar/datasets/audioset_music_mood/labels_eval.csv"
}


if __name__ == "__main__":
    print("\n")
    # suppress warnings:
    warnings.filterwarnings("ignore")

    # get music mood labels label IDs:
    print("Getting music mood label IDs...")

    # load ontology dictionary:
    with open(ONTOLOGY_FILE, "r") as json_file:
        ontology = json.load(json_file)
    
    # extract label IDs from ontology:
    mood_label_ids = {}
    for entry in ontology:
        if entry["name"] in MOOD_LABEL_NAMES:
            if entry["name"] in mood_label_ids.values():
                raise RuntimeError("'{}' already found in ontology.".format(entry["name"]))
            else:
                mood_label_ids[entry["id"]] = entry["name"]
    

    # load metadata:
    print("\nLoading metadata...")
    metadata_dfs = {}
    col_names = ["youtube_id", "start_seconds", "end_seconds", "positive_labels"]
    for subset in data_subsets:
        metadata_dfs[subset] = pd.read_csv(METADATA_FILES[subset], sep=", ", names=col_names, header=None, skiprows=[0, 1, 2])
    

    # extract audio file names:
    print("\nExtracting audio file names...")
    audio_file_names = {}
    for subset in data_subsets:
        subset_root = os.path.join(DATA_ROOT, subset)
        audio_file_names[subset] = [name for name in os.listdir(subset_root) if os.path.isfile(os.path.join(subset_root, name))]
    

    # extract labels of audio files:
    print()
    label_dfs = {}
    for subset in data_subsets:
        file_names = []
        mood_labels = []
        n_bad_files = 0     # number of audio files with not exactly 1 music mood label
        print()
        for file_name in tqdm.tqdm(audio_file_names[subset], total=len(audio_file_names[subset]), desc="Extracting labels for {} set".format(subset)):
            # extract youtube ID from file name:
            youtube_id_start_time = file_name.replace(".wav", "")
            assert len(youtube_id_start_time) == len(file_name) - len(".wav"), "Error with removing '.wav'."
            start_time = "_" + youtube_id_start_time.split("_")[-1]
            youtube_id = youtube_id_start_time.replace(start_time, "")
            assert len(youtube_id) == len(youtube_id_start_time) - len(start_time), "Error with removing start_time."

            # get label ids by querying metadata with youtube id:
            label_ids = metadata_dfs[subset]["positive_labels"][metadata_dfs[subset]["youtube_id"] == youtube_id]

            # convert format of labels ids:
            label_ids = label_ids.reset_index(drop=True)
            label_ids = label_ids[0]
            # remove leading/trailing quotation marks:
            orig_len = len(label_ids)
            label_ids = label_ids.strip('"')
            assert len(label_ids) == orig_len - 2, "Error with removing leading/trailing quotation marks from labels string."
            # convert to list:
            label_ids = label_ids.split(",")
            assert type(label_ids) == list, "Error with converting labels string to list."

            # extract music mood label (should be exactly 1):
            mood_label = None
            n_mood_labels = 0
            for label_id in label_ids:
                if label_id in mood_label_ids.keys():
                    mood_label = mood_label_ids[label_id]
                    n_mood_labels += 1
            # only keep audio files with exactly 1 music mood label:
            if n_mood_labels == 1:
                file_names.append(subset + "/" + file_name)
                mood_labels.append(mood_label)
            else:
                n_bad_files += 1
        
        print("Number of audio files with not exactly 1 music mood label: {}".format(n_bad_files))

        # save as dataframe:
        label_dfs[subset] = pd.DataFrame(
            data={
                "file_name": file_names,
                "label": mood_labels
            }
        )
        # save to file:
        label_dfs[subset].to_csv(label_files[subset], index=False)
    

    print("\n")

