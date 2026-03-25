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

#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace cudf::detail::row::equality {
struct preprocessed_table;
}

namespace spark_rapids_jni {

/**
 * @file
 * @brief Fused ROLLUP aggregation API (libcudf `groupby`-style), for JNI to call directly in C++.
 *
 * `rollup_spec` identifies which columns of `keys` participate in the rollup hierarchy (subtotals).
 * Key columns of `keys` whose indices are not listed behave as ordinary GROUP BY keys present in
 * every grouping set. Listed indices follow rollup ordering (e.g. Spark `ROLLUP(c0, c1)` →
 * `{0, 1}` when those are the only key columns).
 */

/// @brief Same request shape as `cudf::groupby::aggregate`.
using rollup_aggregation_request = cudf::groupby::aggregation_request;

/// @brief Same result shape as `cudf::groupby::aggregate`.
using rollup_aggregation_result = cudf::groupby::aggregation_result;

/**
 * @brief Selects which `keys` columns are rolled up (and in which order within the hierarchy).
 *
 * Indices are zero-based positions in the `keys` table passed to `rollup`. Implementations may require
 * additional invariants (e.g. uniqueness); those will be documented with the kernel.
 */
struct rollup_spec {
  std::vector<cudf::size_type> rolled_up_key_column_indices{};
};

/**
 * @brief Groups by `keys`, computes ROLLUP subtotals, and runs aggregations analogous to
 * `cudf::groupby::groupby`.
 *
 * Lifetime: does not own `keys`; the caller must keep key data alive for the duration of this
 * object and any `aggregate` call, matching `cudf::groupby::groupby`.
 */
class rollup {
 public:
  rollup() = delete;
  ~rollup();
  rollup(rollup const&)            = delete;
  rollup(rollup&&)                 = delete;
  rollup& operator=(rollup const&) = delete;
  rollup& operator=(rollup&&)      = delete;

  /**
   * @param keys Table whose rows are grouping keys (plain keys plus rollup key columns).
   * @param spec Which key columns take part in rollup; see `rollup_spec`.
   * @param null_handling Same as `cudf::groupby::groupby`: `INCLUDE` keeps all input rows (and each
   *        rollup virtual row) in the aggregation; `EXCLUDE` skips virtual rows where any *active*
   *        key column is null for that grouping level. JNI maps `ignoreNullKeys` to `EXCLUDE` /
   *        `INCLUDE` respectively; the Spark plugin uses `INCLUDE` to match
   *        `GroupByOptions.withIgnoreNullKeys(false)`.
   * @param keys_are_sorted Whether `keys` are pre-sorted (same role as groupby).
   * @param column_order If sorted, ascending/descending per key column.
   * @param null_precedence If sorted, null ordering per key column.
   */
  explicit rollup(cudf::table_view const& keys,
                  rollup_spec spec,
                  cudf::null_policy null_handling                      = cudf::null_policy::EXCLUDE,
                  cudf::sorted keys_are_sorted                         = cudf::sorted::NO,
                  std::vector<cudf::order> const& column_order         = {},
                  std::vector<cudf::null_order> const& null_precedence = {});

  /**
   * @brief Runs grouped aggregations across rollup grouping sets (fused Spark ROLLUP partial path).
   *
   * Appends one INT64 column after the key columns: Spark `spark_grouping_id` =
   * `(1 << grouping_level) - 1` with `grouping_level = virtual_index % num_levels` (same literals as
   * GpuExpand / fused expand+agg kernels in Ferdinand).
   *
   * @return Key columns, then `spark_grouping_id`, then one `rollup_aggregation_result` per
   *         request (in request order).
   */
  std::pair<std::unique_ptr<cudf::table>, std::vector<rollup_aggregation_result>> aggregate(
    cudf::host_span<rollup_aggregation_request const> requests,
    rmm::cuda_stream_view stream      = cudf::get_default_stream(),
    rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  [[nodiscard]] cudf::table_view const& keys() const noexcept { return keys_; }
  [[nodiscard]] rollup_spec const& spec() const noexcept { return spec_; }

 private:
  cudf::table_view keys_;
  rollup_spec spec_;
  cudf::null_policy null_handling_;
  cudf::sorted keys_are_sorted_;
  std::vector<cudf::order> column_order_;
  std::vector<cudf::null_order> null_precedence_;

  /**
   * libcudf hash groupby precomputes this via
   * `cudf::detail::row::equality::preprocessed_table::create(keys, stream)` (see
   * `cudf::groupby::detail::hash::dispatch_groupby`). Used for row hashing and row equality on
   * unsorted keys; unused for the sorted path until implemented.
   */
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_keys_{};
};

}  // namespace spark_rapids_jni
