import os, numpy as np, subprocess
import torch, torchaudio
from typing import Tuple, Optional
from torch.utils.data import Dataset
from torch.hub import download_url_to_file

torchaudio.set_audio_backend("soundfile")
from torch import Tensor, FloatTensor
from zipfile import ZipFile

FOLDER_IN_ARCHIVE = ""
_CHECKSUMS = {
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003": "",
    "http://mi.soi.city.ac.uk/datasets/magnatagatune/annotations_final.csv": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/binary.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/tags.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/test.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/train.npy": "",
    "https://github.com/minzwon/sota-music-tagging-models/raw/master/split/mtat/valid.npy": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/train_gt_mtt.tsv": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/val_gt_mtt.tsv": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/test_gt_mtt.tsv": "",
    "https://github.com/jordipons/musicnn-training/raw/master/data/index/mtt/index_mtt.tsv": "",
}


def get_file_list(root, subset, split):
    if subset == "train":
        if split == "pons2017":
            fl = open(os.path.join(root, "train_gt_mtt.tsv")).read().splitlines()
        else:
            fl = np.load(os.path.join(root, "train.npy"))
    elif subset == "valid":
        if split == "pons2017":
            fl = open(os.path.join(root, "val_gt_mtt.tsv")).read().splitlines()
        else:
            fl = np.load(os.path.join(root, "valid.npy"))
    else:
        if split == "pons2017":
            fl = open(os.path.join(root, "test_gt_mtt.tsv")).read().splitlines()
        else:
            fl = np.load(os.path.join(root, "test.npy"))

    if split == "pons2017":
        binary = {}
        index = open(os.path.join(root, "index_mtt.tsv")).read().splitlines()
        fp_dict = {}
        for i in index:
            clip_id, fp = i.split("\t")
            fp_dict[clip_id] = fp

        for idx, f in enumerate(fl):
            clip_id, label = f.split("\t")
            fl[idx] = "{}\t{}".format(clip_id, fp_dict[clip_id])
            clip_id = int(clip_id)
            binary[clip_id] = eval(label)
    else:
        binary = np.load(os.path.join(root, "binary.npy"))

    return fl, binary


class MTAT(Dataset):
    """Create a Dataset for MagnaTagATune.
    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        folder_in_archive (str, optional): The top-level directory of the dataset.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """

    _ext_audio = ".wav"

    def __init__(
        self,
        root: str,
        folder_in_archive: Optional[str] = FOLDER_IN_ARCHIVE,
        download: Optional[bool] = False,
        subset: Optional[str] = None,
        split: Optional[str] = "pons2017",
        sr: int = 22050,
        duration: int = 30,
    ) -> None:

        super(MTAT, self).__init__()
        self.sr = sr
        self.root = root
        self.folder_in_archive = folder_in_archive
        self.download = download
        self.subset = subset
        self.split = split
        self.duration = duration

        assert subset is None or subset in ["train", "valid", "test"], (
            "When `subset` not None, it must take a value from " + "{'train', 'valid', 'test'}."
        )

        self._path = os.path.join(root, folder_in_archive)
        self.label_list = np.load(self._path + "tags.npy")

        if download:
            if not os.path.isdir(self._path):
                os.makedirs(self._path)

            zip_files = []
            for url, checksum in _CHECKSUMS.items():
                target_fn = os.path.basename(url)
                target_fp = os.path.join(self._path, target_fn)
                if ".zip" in target_fp:
                    zip_files.append(target_fp)

                if not os.path.exists(target_fp):
                    download_url_to_file(
                        url,
                        self._path,
                        filename=target_fn,
                        hash_value=checksum,
                        hash_type="md5",
                    )

            if not os.path.exists(
                os.path.join(
                    self._path,
                    "f",
                    "american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.mp3",
                )
            ):
                merged_zip = os.path.join(self._path, "mp3.zip")
                print("Merging zip files...")
                with open(merged_zip, "wb") as f:
                    for filename in zip_files:
                        with open(filename, "rb") as g:
                            f.write(g.read())

                # extract_archive(merged_zip)
                with ZipFile(merged_zip, "r") as f:
                    f.extractall()

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found. Please use `download=True` to download it.")

        self.fl, self.binary = get_file_list(self._path, self.subset, self.split)
        self.n_classes = len(self.label_list)

        self.label2idx = {}
        for idx, label in enumerate(self.label_list):
            self.label2idx[label] = idx

    def file_path(self, n: int) -> str:
        _, fp = self.fl[n].split("\t")
        return os.path.join(self._path, fp)

    def target_file_path(self, n: int) -> str:
        fp = self.file_path(n)
        file_basename, _ = os.path.splitext(fp)
        return file_basename + self._ext_audio

    def preprocess(self, n: int, sample_rate: int):
        fp = self.file_path(n)
        tfp = self.target_file_path(n)
        if not os.path.exists(tfp):
            p = subprocess.Popen(
                ["ffmpeg", "-i", fp, "-ar", str(sample_rate), tfp, "-loglevel", "quiet"]
            )
            p.wait()

    def segment_audio_sample(self):
        audio_len = audio.shape[0]
        segment_len = int(self.duration * self.sr)

        if audio_len < segment_len:
            audio = torch.cat([audio, torch.zeros(segment_len - audio_len)])

        if self.subset == "train":
            # randomly crop to target clip length:
            start_idx = np.random.randint(low=0, high=audio_len - self.duration + 1)
            end_idx = start_idx + self.duration
            audio = audio[start_idx:end_idx]
            assert audio.size(dim=0) == self.duration, "Error with cropping audio clip."

        else:
            # split audio clip into chunks:
            step = int(np.around((1 - self.overlap_ratio) * self.duration))
            audio_chunks = audio.unfold(dimension=0, size=self.duration, step=step)
            # sanity check shape:
            assert (
                len(tuple(audio_chunks.size())) == 2 and audio_chunks.size(dim=-1) == self.duration
            ), "Error with shape of chunked audio clip."

        return audio

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        clip_id, _ = self.fl[n].split("\t")
        label = self.binary[int(clip_id)]

        filepath = self.file_path(n).replace(".mp3", ".wav")
        audio, sr = torchaudio.load(filepath)
        if sr != self.sr:
            resample = torchaudio.transforms.Resample(sr, self.sr)
            audio = resample(audio)
        print(audio.shape)
        audio = self.segment_audio_sample(audio.squeeze(0))
        return audio, FloatTensor(label)

    def __len__(self) -> int:
        return len(self.fl)


if __name__ == "__main__":
    dataset = MTAT(
        root="/data/avramidi/VCMR/data/magnatagatune", download=True, subset="valid", sr=16000
    )
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
