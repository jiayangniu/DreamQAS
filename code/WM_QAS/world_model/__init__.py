# world_model package (current method).
# Only `networks.MLP` is live — used by agent/wm_policy.py and agent/baseline_policy.py.
# The RSSM / DreamerV3 modules (rssm.py, encoders.py, train_wm.py) were the earlier
# learned-dynamics line; they are archived at code/_archive/rssm_line/world_model/.
# Import networks explicitly:  from world_model.networks import MLP
