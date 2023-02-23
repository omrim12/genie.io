## genie.io
![genie image](genie_logo.png)
### Pre-requisites:
- download [food-101](https://www.kaggle.com/datasets/kmader/food41) images datasets to this path
### Get started with genie running following commands:
- create a virtualenv:
```bash
python -m venv venv && source venv/bin/activate
```
- install requirements:
```bash
pip install -r requirements.txt
```
*Note: app requires python version>=3.10*
- run genie in CLI mode:
```bash
python genie_driver.py --cli-mode
```
- run genie in REST API mode:
```bash
python genie_driver.py --api-mode
```
- train a new genie CNN model:
```bash
python genie_driver.py --new-genie
```
