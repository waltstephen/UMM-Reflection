# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import pyarrow.parquet as pq

from ..distributed_iterable_dataset import DistributedIterableDataset
from ..parquet_utils import get_parquet_data_paths, init_arrow_pf_fs


class InterleavedBaseIterableDataset(DistributedIterableDataset):

    def _init_data(self):
        data = {
            'sequence_plan': [],
            'text_ids_list': [],
            'image_tensor_list': [],
            'num_tokens': 0,
        }
        return data

    def _add_text(self, data, text, need_loss, enable_cfg=True, ce_loss_weights=None):
        text_ids = self.tokenizer.encode(text)
        plan = {
            'type': 'text',
            'enable_cfg': int(enable_cfg),
            'loss': int(need_loss),
            'special_token_loss': 0,
            'special_token_label': None,
        }
        if ce_loss_weights is not None:
            weights = [float(weight) for weight in ce_loss_weights]
            # CE labels are text_ids + EOS, so the optional override is label-aligned.
            if len(weights) == len(text_ids):
                weights.append(1.0)
            if len(weights) != len(text_ids) + 1:
                raise ValueError(
                    f"ce_loss_weights length {len(weights)} does not match "
                    f"text CE labels length {len(text_ids) + 1}"
                )
            plan['ce_loss_weights'] = weights
        data['num_tokens'] += len(text_ids)
        data['text_ids_list'].append(text_ids)
        data['sequence_plan'].append(plan)
        return data

    def _add_image(self, data, image, need_loss, need_vae, need_vit, enable_cfg=True):
        assert need_loss or need_vae or need_vit

        if need_loss:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor)

        if need_vae:
            data['sequence_plan'].append(
                {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                }
            )

            image_tensor = self.transform(image)
            height, width = image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.transform.stride ** 2
            data['image_tensor_list'].append(image_tensor.clone())

        if need_vit:
            data['sequence_plan'].append(
                {
                    'type': 'vit_image',
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0,
                    'special_token_loss': 0,
                    'special_token_label': None,
                },
            )
            vit_image_tensor = self.vit_transform(image)
            height, width = vit_image_tensor.shape[1:]
            data['num_tokens'] += width * height // self.vit_transform.stride ** 2
            data['image_tensor_list'].append(vit_image_tensor)

        return data

    def _add_video(self, data, frames, frame_indexes, need_loss, need_vae, enable_cfg=True):
        assert int(need_loss) + int(need_vae) == 1

        if need_loss:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': 0, 
                    'loss': 1, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        elif need_vae:
            for idx, (image, frame_idx) in enumerate(zip(frames, frame_indexes)):
                current_sequence_plan = {
                    'type': 'vae_image', 
                    'enable_cfg': int(enable_cfg), 
                    'loss': 0, 
                    'special_token_loss': 0,
                    'special_token_label': None,
                    'split_start': idx == 0,
                    'split_end': idx == len(frames) - 1,
                }
                if idx < len(frame_indexes) - 1:
                    current_sequence_plan['frame_delta'] = frame_indexes[idx + 1] - frame_idx
                data['sequence_plan'].append(current_sequence_plan)
                image_tensor = self.transform(image)
                height, width = image_tensor.shape[1:]
                data['image_tensor_list'].append(image_tensor)
                data['num_tokens'] += width * height // self.transform.stride ** 2

        return data


class ParquetStandardIterableDataset(DistributedIterableDataset):
    _resume_batch_size = 64

    def __init__(
        self, dataset_name, transform, tokenizer, vit_transform, 
        data_dir_list, num_used_data, parquet_info,
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        vit_transform: input transform for vit model.
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.data_paths = self.get_data_paths(data_dir_list, num_used_data, parquet_info)
        self.set_epoch()

    def get_data_paths(self, data_dir_list, num_used_data, parquet_info):
        row_groups = []
        for data_dir, num_data_path in zip(data_dir_list, num_used_data):
            data_paths = get_parquet_data_paths([data_dir], [num_data_path])
            for data_path in data_paths:
                if data_path in parquet_info.keys():
                    num_row_groups = parquet_info[data_path]['num_row_groups']
                    for rg_idx in range(num_row_groups):
                        row_groups.append((data_path, rg_idx))
        return row_groups

    def parse_row(self, row):
        raise NotImplementedError

    def parse_row_from_source(self, row, data_path, row_group_id, row_idx):
        """Source-aware opt-in hook; existing readers keep parse_row behavior."""
        return self.parse_row(row)

    def get_read_columns(self, data_path):
        """Opt-in parquet projection hook; legacy readers read all columns."""
        return None

    def _resume_position(self, worker_id):
        if self.data_status is None:
            return 0, 0

        worker_status = None
        if hasattr(self.data_status, "get"):
            worker_status = self.data_status.get(worker_id)
            if worker_status is None:
                worker_status = self.data_status.get(str(worker_id))
        elif worker_id < len(self.data_status):
            worker_status = self.data_status[worker_id]

        if worker_status is None:
            return 0, 0

        return int(worker_status[0]), int(worker_status[1]) + 1

    def _iter_row_group_rows(
        self,
        parquet_file,
        row_group_id,
        row_start_id,
        columns=None,
    ):
        num_rows = parquet_file.metadata.row_group(row_group_id).num_rows
        if row_start_id >= num_rows:
            return

        # Do not materialize rows that were already consumed before checkpointing.
        # Arrow batches still preserve the original row order after the cheap skip.
        next_row_idx = 0
        for batch in parquet_file.iter_batches(
            row_groups=[row_group_id],
            batch_size=self._resume_batch_size,
            columns=columns,
        ):
            batch_num_rows = batch.num_rows
            batch_end_idx = next_row_idx + batch_num_rows
            if batch_end_idx <= row_start_id:
                next_row_idx = batch_end_idx
                continue

            offset = max(row_start_id - next_row_idx, 0)
            if offset:
                batch = batch.slice(offset)
            batch_start_idx = next_row_idx + offset
            df = batch.to_pandas()
            for local_idx, (_, row) in enumerate(df.iterrows()):
                yield batch_start_idx + local_idx, row

            next_row_idx = batch_end_idx

    def __iter__(self):
        file_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        global_row_group_start_id, row_start_id = self._resume_position(worker_id)
        if global_row_group_start_id >= len(file_paths_per_worker):
            global_row_group_start_id = 0
            row_start_id = 0

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at global_rg#{global_row_group_start_id}, row#{row_start_id}"
        )

        while True:
            file_paths_per_worker_ = file_paths_per_worker[global_row_group_start_id:]
            for global_row_group_idx, (parquet_file_path, row_group_id) in enumerate(
                file_paths_per_worker_, start=global_row_group_start_id
            ):
                fs = init_arrow_pf_fs(parquet_file_path)
                with fs.open_input_file(parquet_file_path) as f:
                    try:
                        fr = pq.ParquetFile(f)
                    except Exception as e:
                        if getattr(self, "fail_fast", False):
                            raise
                        print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                        continue

                    try:
                        read_columns = self.get_read_columns(
                            parquet_file_path
                        )
                        for row_idx, row in self._iter_row_group_rows(
                            fr,
                            row_group_id,
                            row_start_id,
                            columns=read_columns,
                        ):
                            try:
                                data = self.parse_row_from_source(
                                    row,
                                    parquet_file_path,
                                    row_group_id,
                                    row_idx,
                                )
                                if len(data) == 0:
                                    continue
                                sample_metadata = data.pop("_sample_metadata", {})
                                data_indexes = {
                                    "data_indexes": [global_row_group_idx, row_idx],
                                    "worker_id": worker_id,
                                    "dataset_name": self.dataset_name,
                                }
                                uid = str(row.get("uid", "") or "")
                                if uid:
                                    run = str(row.get("run", "") or "")
                                    data_indexes.update(
                                        {
                                            "uid": uid,
                                            "run": run,
                                            "task": str(row.get("task", "") or ""),
                                            "source_split": "new" if run == "unit_text_v2" else "replay",
                                        }
                                    )
                                data_indexes.update(sample_metadata)
                                data['data_indexes'] = data_indexes
                            except Exception as e:
                                if getattr(self, "fail_fast", False):
                                    raise
                                print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')
                                continue
                            yield data
                    except Exception as e:
                        if getattr(self, "fail_fast", False):
                            raise
                        print(f'Error {e} in rg#{row_group_id}, {parquet_file_path}')

                    row_start_id = 0
            global_row_group_start_id = 0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
