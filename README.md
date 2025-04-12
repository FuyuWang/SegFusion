# SFD


### Setup ###
* Download the SFD source code 

* Create virtual environment through anaconda
```
conda create --name SFDEnv python=3.8
conda activate SFDEnv
```
* Install packages
   
```
cd SFD
pip install -r requirements.txt
```

* Install [Timeloop](https://timeloop.csail.mit.edu/timeloop)

### Run SFD on NVDLA ###

```
cd nvdla_cnn
sh ./run.sh
```

