# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import os

from .interleave_datasets import UnifiedEditIterableDataset
from .interleave_datasets.base_ema_decoupled_payload_dataset import (
    BasePromptOnlyAnchorMSEIterableDataset,
)
from .interleave_datasets.clean29529_multiround_transition_dataset import (
    Clean29529MultiroundControllerIterableDataset,
    Clean29529PenultimateVerifierStateIterableDataset,
    Clean29529TransitionMSEIterableDataset,
    Clean29529VerifierStateIterableDataset,
)
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'base_prompt_only_anchor_mse': BasePromptOnlyAnchorMSEIterableDataset,
    'clean29529_multiround_controller': Clean29529MultiroundControllerIterableDataset,
    'clean29529_transition_mse': Clean29529TransitionMSEIterableDataset,
    'clean29529_verifier_state': Clean29529VerifierStateIterableDataset,
    'clean29529_penultimate_verifier_state': Clean29529PenultimateVerifierStateIterableDataset,
}


# Reflection SFT data lives under $UNIFY_RL_DATA_ROOT/sft (see README):
#   rows/                  output of scripts/prepare_clean29529_multiround_transition_data.py
#   anchor/parquet/        base-BAGEL prompt-only anchor images (1,265 allowlisted rows)
_SFT_ROOT = os.path.join(os.environ.get('UNIFY_RL_DATA_ROOT', 'data'), 'sft')


def _parquet_entry(subdir, num_files, num_total_samples):
    data_dir = os.path.join(_SFT_ROOT, subdir)
    return {
        'data_dir': data_dir,
        'num_files': num_files,
        'num_total_samples': num_total_samples,
        'parquet_info_path': os.path.join(data_dir, 'parquet_info.json'),
    }


DATASET_INFO = {
    't2i_pretrain': {
        't2i': {
            'data_dir': 'your_data_path/bagel_example/t2i', # path of the parquet files
            'num_files': 10, # number of data units to be sharded across all ranks and workers
            'num_total_samples': 1000, # number of total samples in the dataset
        },
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': 'your_data_path/bagel_example/editing/seedxedit_multi',
            'num_files': 10,
            'num_total_samples': 1000,
            "parquet_info_path": 'your_data_path/bagel_example/editing/parquet_info/seedxedit_multi_nas.json', # information of the parquet files
        },
    },
    'vlm_sft': {
        'llava_ov': {
            'data_dir': 'your_data_path/bagel_example/vlm/images',
            'jsonl_path': 'your_data_path/bagel_example/vlm/llava_ov_si.jsonl',
            'num_total_samples': 1000
        },
    },
    'base_prompt_only_anchor_mse': {
        'reflection_t2i': _parquet_entry('anchor/parquet', 96, 1_265),
    },
    'clean29529_multiround_controller': {
        'clean29529_multiround_controller': _parquet_entry('rows/controller_rows_100', 100, 29_529),
    },
    'clean29529_transition_mse': {
        'clean29529_transition_mse': _parquet_entry('rows/transition_rows_96', 96, 58_020),
    },
    'clean29529_verifier_state': {
        'clean29529_verifier_state': _parquet_entry('rows/verifier_rows_96', 96, 58_020),
    },
}
DATASET_INFO['clean29529_penultimate_verifier_state'] = {
    'clean29529_penultimate_verifier_state': dict(
        DATASET_INFO['clean29529_verifier_state']['clean29529_verifier_state']
    )
}
