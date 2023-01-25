## Ginny.io
### Pre-requisites:
- download [food-101](https://www.kaggle.com/datasets/kmader/food41) images datasets to this path
### Get started with ginny running following commands:
- create a virtualenv:
```bash
python -m venv venv && source venv/bin/activate
```
- install requirements:
```bash
pip install -r requirements.txt
```
*Note: app requires python version>=3.10*
- run ginny in CLI mode:
```bash
python ginny_driver.py --cli-mode
```
- train a new ginny CNN model:
```bash
python ginny_driver.py --new-ginny
```