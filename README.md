## SoccerNet amateur model configs
---

A simple model with plain and transfer learning capabilities SoccerNet for amateur dataset

how to install OSL-Action can be found [here](https://github.com/OpenSportsLab/OSL-ActionSpotting/blob/main/docs/install.md)

to run simple model:
https://github.com/OpenSportsLab/OSL-ActionSpotting/blob/main/docs/usage.md

To run in SoccerNet-Amateur dataset, copy configs file to OSL-action repo, like:

```bash
cp -rf configs/contextawarelossfunction/* <dir of OSL action repo>/configs/contextawarelossfunction/
```

then run the dataset amateur download:

```bash
./scripts/download.sh
# or if set a directory specific

./scripts/download.sh /opt/datasets/
```

finally, run:

```bash
./train.sh configs/contextawarelossfunction/json_soccernet_calf_resnetpca512_amateur_model.py 200 /opt/datasets/ 
```