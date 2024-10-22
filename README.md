
# DCARA


## 1. Usage

### 1.1 Preparation

+ Create and activate virtual environment:

```
python3 -m venv ~/DCARA-env
source ~/DCARA-env/bin/activate
```

+ Clone the repository and navigate to new directory:

```
git clone https://github.com/Dylan-1978/DCARA
cd ./DCARA
```

+ Install the requirements:

```
pip install -r requirements.txt
```

+ Download and extract the [Kvasir-SEG](https://datasets.simula.no/downloads/kvasir-seg.zip) and the [CVC-ClinicDB](https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar?dl=0) datasets.

+ Download the [PVTv2-B3](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth) weights to `./`

### 1.2 Training

Train DCARA on the train split of a dataset:

```
python train.py 
```

### 1.3. Notice
更多的工作细节将在后续补充