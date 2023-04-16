

# CLIMuR: **C**ontrastive **L**anguage-augmented **I**mage-to-**MU**sic **R**etrieval

This repository contains the code for CLIMuR: Contrastive Language-augmented Image-to-MUsic Retrieval. The focus is on augmenting image queries with their natural language captions (i.e., using multimodal image+text queries) for emotion-aligned cross-modal music retrieval.



## Project Structure

```
CLIMuR/
├── climur/               # main directory for CLIMuR
│   ├── dataloaders/         # PyTorch Dataset classes
│   ├── losses/              # PyTorch loss functions
│   ├── models/              # PyTorch Module classes
│   ├── scripts/             # training and evaluation scripts
│   ├── trainers/            # PyTorch Lightning LightningModule classes
│   ├── utils/               # utility functions
├── configs/              # configuration files for training and evaluation
├── data_prep/            # data preparation scripts
├── tests/                # test scripts
```

