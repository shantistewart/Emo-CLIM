"""Script to extract wav files from mp4 files."""


import os
import subprocess
import tqdm


# data paths:
SOURCE_DIR = "/proj/systewar/datasets/audioset_music_mood/balanced_train_preprocess/balanced_train_mp4"
TARGET_DIR = "/proj/systewar/datasets/audioset_music_mood/balanced_train_preprocess/balanced_train_wav_full"
# target sampling rate:
SAMPLE_RATE = 16000


if __name__ == "__main__":
    print("\n")

    # create target directory:
    os.makedirs(TARGET_DIR, exist_ok=True)

    # extract mp4 file names:
    mp4_file_names = [name for name in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, name))]
    print("Number of mp4 files: {}".format(len(mp4_file_names)))

    # extract wav files from mp4 files:
    for mp4_file_name in tqdm.tqdm(mp4_file_names, total=len(mp4_file_names), desc="Extracting wav files..."):
        assert mp4_file_name.endswith(".mp4"), "File path does not end in '.mp4'"

        # create .wav file name:
        file_name = mp4_file_name.replace(".mp4", "")
        wav_file_name = file_name + ".wav"

        # call ffmpeg command:
        command = f"ffmpeg -i {os.path.join(SOURCE_DIR, mp4_file_name)} -ab 160k -ac 1 -ar {SAMPLE_RATE} -loglevel warning -vn {os.path.join(TARGET_DIR, wav_file_name)}"
        subprocess.call(command, shell=True)
    

    print("\n")

