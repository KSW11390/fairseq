# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
torch.hub entry point for s3prl custom upstream.

s3prl loads this file via --upstream_expert flag:
    python run_downstream.py \
        --upstream custom \
        --upstream_expert /path/to/hubconf.py \
        --upstream_ckpt /path/to/checkpoint.pt \
        --downstream <TASK> ...
"""

import os
import sys

# Ensure the parent directory is on the path so `expert` can be imported
sys.path.insert(0, os.path.dirname(__file__))


def dicehubert_local(ckpt, *args, **kwargs):
    """Load a DICEHuBERT model from a local checkpoint path."""
    from expert import UpstreamExpert

    return UpstreamExpert(ckpt, **kwargs)


def dicehubert(ckpt, *args, **kwargs):
    """Alias for dicehubert_local."""
    return dicehubert_local(ckpt, *args, **kwargs)
