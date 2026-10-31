"""cvtrain: train YOLO-family detectors on a PC and package them for the
Jetson Nano.

Every route (training/quickstart.py, the NNI search in training/nni/) ends
in the same place: a model folder that follows the model contract
(docs/model-contract.md), ready for host/deploy_model.sh.

Import compat first (paths.py does): it patches filelock before anything
imports ultralytics.
"""
