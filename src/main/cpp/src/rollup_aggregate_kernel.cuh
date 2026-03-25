/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "rollup_hash.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

namespace spark_rapids_jni::detail {

/**
 * @brief Grid is one block per SM; each block covers `rows_per_block` consecutive input rows.
 *
 * Threads stride over that row range, then the inner loop runs rollup grouping levels (group id).
 * For each virtual key: `insert_and_find` (sparse slot = canonical virtual index), then for each
 * value column apply the same `dispatch_type_and_aggregation` path as
 * `cudf::groupby::detail::hash::compute_single_pass_aggs_sparse_output_fn`.
 *
 * Virtual index: `virtual_ix = input_row * num_levels + group_id`.
 */
template <typename SetRef>
CUDF_KERNEL void rollup_aggregate_insert_find_kernel(cudf::size_type num_input_rows,
                                                     cudf::size_type rows_per_block,
                                                     cudf::size_type num_levels,
                                                     bool exclude_null_keys,
                                                     cudf::table_device_view keys_dview,
                                                     cudf::size_type const* rolled_rank,
                                                     cudf::size_type num_rolled,
                                                     cudf::table_device_view input_values,
                                                     cudf::mutable_table_device_view output_values,
                                                     cudf::aggregation::Kind const* d_agg_kinds,
                                                     SetRef set_ref)
{
  auto const row_base = static_cast<cudf::size_type>(blockIdx.x) * rows_per_block;

  for (cudf::size_type local = threadIdx.x; local < rows_per_block; local += blockDim.x) {
    cudf::size_type const row = row_base + local;
    if (row >= num_input_rows) { continue; }

    for (cudf::size_type group_id = 0; group_id < num_levels; ++group_id) {
      cudf::size_type const virtual_ix = row * num_levels + group_id;
      if (exclude_null_keys &&
          rollup_skip_virtual_row_for_null_exclude(
            keys_dview, rolled_rank, num_rolled, num_levels, virtual_ix)) {
        continue;
      }

      auto const target_row_idx = *set_ref.insert_and_find(virtual_ix).first;

      for (cudf::size_type col_idx = 0; col_idx < input_values.num_columns(); ++col_idx) {
        auto const& source_col = input_values.column(col_idx);
        auto& target_col       = output_values.column(col_idx);
        cudf::detail::dispatch_type_and_aggregation(source_col.type(),
                                                      d_agg_kinds[col_idx],
                                                      cudf::detail::element_aggregator{},
                                                      target_col,
                                                      target_row_idx,
                                                      source_col,
                                                      row);
      }
    }
  }
}

}  // namespace spark_rapids_jni::detail
