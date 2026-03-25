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

#include "rollup.hpp"
#include "rollup_aggregate_kernel.cuh"
#include "rollup_output_keys.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/row_operator/preprocessed_table.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <common/utils.hpp>
#include <hash/extract_single_pass_aggs.hpp>
#include <hash/hash_compound_agg_finalizer.hpp>
#include <hash/output_utils.hpp>

#include <cuco/static_set.cuh>

#include <cuda_runtime.h>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] std::pair<rmm::device_buffer, cudf::bitmask_type const*> rollup_keys_row_bitmask(
  cudf::table_view const& keys,
  bool exclude_null_keys,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (not exclude_null_keys or not cudf::has_nulls(keys)) {
    return {rmm::device_buffer{}, nullptr};
  }
  if (keys.num_columns() == 1) {
    auto const& keys_col = keys.column(0);
    if (keys_col.offset() == 0) { return {rmm::device_buffer{}, keys_col.null_mask()}; }
    auto null_mask_data  = cudf::copy_bitmask(keys_col, stream);
    auto const null_mask = static_cast<cudf::bitmask_type const*>(null_mask_data.data());
    return {std::move(null_mask_data), null_mask};
  }
  auto [null_mask_data, null_count] = cudf::bitmask_and(keys, stream, mr);
  if (null_count == 0) { return {rmm::device_buffer{}, nullptr}; }
  auto const null_mask = static_cast<cudf::bitmask_type const*>(null_mask_data.data());
  return {std::move(null_mask_data), null_mask};
}

[[nodiscard]] std::unique_ptr<cudf::table> materialize_rollup_output_keys(
  cudf::table_view const& keys_table,
  rmm::device_uvector<cudf::size_type> const& d_unique_virtual,
  cudf::size_type num_levels,
  cudf::size_type num_rolled,
  std::vector<cudf::size_type> const& h_rolled_rank,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_unique = static_cast<cudf::size_type>(d_unique_virtual.size());
  if (num_unique == 0) { return cudf::empty_like(keys_table); }

  rmm::device_uvector<cudf::size_type> d_input_row(num_unique, stream);
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    d_unique_virtual.begin(),
    d_unique_virtual.end(),
    d_input_row.begin(),
    [num_levels] __device__(cudf::size_type const v) { return v / num_levels; });

  auto const gather_span =
    cudf::device_span<cudf::size_type const>{d_input_row.data(), d_input_row.size()};

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(static_cast<std::size_t>(keys_table.num_columns()));

  constexpr int key_k_threads = 256;
  int const key_k_grid =
    static_cast<int>(cudf::util::div_rounding_up_safe(num_unique, key_k_threads));

  for (cudf::size_type c = 0; c < keys_table.num_columns(); ++c) {
    auto gathered_tbl = cudf::detail::gather(
      cudf::table_view{{keys_table.column(c)}},
      gather_span,
      cudf::out_of_bounds_policy::DONT_CHECK,
      cudf::detail::negative_index_policy::NOT_ALLOWED,
      stream,
      mr);
    auto released        = gathered_tbl->release();
    auto gathered        = std::move(released.front());
    auto active          = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                            num_unique,
                                            cudf::mask_state::ALL_NULL,
                                            stream,
                                            mr);
    auto d_active        = cudf::mutable_column_device_view::create(active->mutable_view(), stream);
    auto const rolled_rc = h_rolled_rank[static_cast<std::size_t>(c)];
    spark_rapids_jni::detail::rollup_active_key_column_kernel<<<key_k_grid, key_k_threads, 0, stream.value()>>>(
      *d_active,
      d_unique_virtual.data(),
      num_unique,
      rolled_rc,
      num_rolled,
      num_levels);
    CUDF_CUDA_TRY(cudaPeekAtLastError());

    auto const [merged_mask, null_count] =
      cudf::bitmask_and(cudf::table_view{{gathered->view(), active->view()}}, stream, mr);
    gathered->set_null_mask(std::move(merged_mask), null_count);
    out_cols.push_back(std::move(gathered));
  }

  return std::make_unique<cudf::table>(std::move(out_cols));
}

[[nodiscard]] std::unique_ptr<cudf::column> make_spark_grouping_id_column(
  rmm::device_uvector<cudf::size_type> const& unique_virtual,
  cudf::size_type num_levels,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_levels > 0 && num_levels < 63,
               "rollup: num_levels out of range for spark_grouping_id");
  auto const n = static_cast<cudf::size_type>(unique_virtual.size());
  if (n == 0) { return cudf::make_empty_column(cudf::type_id::INT64); }
  auto out = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, n, cudf::mask_state::UNALLOCATED, stream, mr);
  auto* d_out = out->mutable_view().data<int64_t>();
  thrust::transform(
    rmm::exec_policy_nosync(stream),
    unique_virtual.begin(),
    unique_virtual.end(),
    d_out,
    [num_levels] __device__(cudf::size_type const v) -> int64_t {
      auto const g = v % num_levels;
      return static_cast<int64_t>((static_cast<std::uint64_t>(1) << g) - 1);
    });
  return out;
}

[[nodiscard]] std::unique_ptr<cudf::table> append_column_after_keys(
  std::unique_ptr<cudf::table> keys_table,
  std::unique_ptr<cudf::column> extra,
  cudf::size_type num_key_columns)
{
  auto cols = keys_table->release();
  CUDF_EXPECTS(static_cast<cudf::size_type>(cols.size()) == num_key_columns,
               "rollup: key table column count mismatch");
  cols.push_back(std::move(extra));
  return std::make_unique<cudf::table>(std::move(cols));
}

}  // namespace

namespace spark_rapids_jni::detail {

[[nodiscard]] inline cudf::size_type device_multiprocessor_count()
{
  int device = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  int num_sms = 0;
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  return static_cast<cudf::size_type>(num_sms);
}

}  // namespace spark_rapids_jni::detail

namespace spark_rapids_jni {

rollup::~rollup() = default;

rollup::rollup(cudf::table_view const& keys,
               rollup_spec spec,
               cudf::null_policy null_handling,
               cudf::sorted keys_are_sorted,
               std::vector<cudf::order> const& column_order,
               std::vector<cudf::null_order> const& null_precedence)
  : keys_(keys),
    spec_(std::move(spec)),
    null_handling_(null_handling),
    keys_are_sorted_(keys_are_sorted),
    column_order_(column_order),
    null_precedence_(null_precedence)
{
  for (auto const idx : spec_.rolled_up_key_column_indices) {
    CUDF_EXPECTS(idx >= 0 && idx < keys_.num_columns(), "rollup key column index out of range");
  }
  std::unordered_set<cudf::size_type> unique_rolled(spec_.rolled_up_key_column_indices.begin(),
                                                    spec_.rolled_up_key_column_indices.end());
  CUDF_EXPECTS(unique_rolled.size() == spec_.rolled_up_key_column_indices.size(),
               "rollup rolled key column indices must be unique");
}

std::pair<std::unique_ptr<cudf::table>, std::vector<rollup_aggregation_result>> rollup::aggregate(
  cudf::host_span<rollup_aggregation_request const> requests,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (keys_are_sorted_ == cudf::sorted::YES) {
    CUDF_FAIL("spark_rapids_jni::rollup sorted-keys path is not implemented");
  }
  CUDF_EXPECTS(not cudf::has_nested_columns(keys_),
               "rollup: nested key columns are not supported in the hash path");
  CUDF_EXPECTS(not requests.empty(), "rollup::aggregate requires at least one aggregation request");
  CUDF_EXPECTS(cudf::groupby::detail::hash::can_use_hash_groupby(requests),
               "rollup: one or more aggregations are not supported on the hash path");

  if (keys_.num_rows() == 0) {
    cudf::groupby::groupby gb(
      keys_, null_handling_, keys_are_sorted_, column_order_, null_precedence_);
    auto empty_out = gb.aggregate(requests, stream, mr);
    auto gid = cudf::make_empty_column(cudf::type_id::INT64);
    empty_out.first =
      append_column_after_keys(std::move(empty_out.first), std::move(gid), keys_.num_columns());
    return empty_out;
  }

  if (!preprocessed_keys_) {
    preprocessed_keys_ = cudf::detail::row::equality::preprocessed_table::create(keys_, stream);
  }

  auto const keys_dview = static_cast<cudf::table_device_view>(*preprocessed_keys_);
  auto const num_input_rows =
    static_cast<cudf::size_type>(keys_.num_rows());  // same row count as preprocessed view
  auto const num_rolled     = static_cast<cudf::size_type>(spec_.rolled_up_key_column_indices.size());
  auto const num_levels     = detail::rollup_num_levels(num_rolled);
  auto const num_virtual    = detail::rollup_virtual_row_count(num_input_rows, num_levels);
  auto const nullate        = cudf::nullate::DYNAMIC{cudf::has_nulls(keys_)};

  std::vector<cudf::size_type> h_rank(static_cast<std::size_t>(keys_.num_columns()), num_rolled);
  for (std::size_t p = 0; p < spec_.rolled_up_key_column_indices.size(); ++p) {
    h_rank[static_cast<std::size_t>(spec_.rolled_up_key_column_indices[p])] =
      static_cast<cudf::size_type>(p);
  }

  auto d_rolled_rank = cudf::detail::make_device_uvector_async(h_rank, stream, mr);

  auto const [values, agg_kinds, aggs, is_agg_intermediate, has_compound_aggs] =
    cudf::groupby::detail::hash::extract_single_pass_aggs(requests, stream);
  CUDF_EXPECTS(values.num_rows() == static_cast<cudf::size_type>(keys_.num_rows()),
               "rollup: values row count must match keys row count");

  auto sparse_agg_results = cudf::groupby::detail::hash::create_results_table(
    num_virtual,
    values,
    agg_kinds,
    std::span<std::int8_t const>{is_agg_intermediate.data(), is_agg_intermediate.size()},
    stream,
    mr);

  auto d_input_values  = cudf::table_device_view::create(values, stream);
  auto d_output_values =
    cudf::mutable_table_device_view::create(sparse_agg_results->mutable_view(), stream);
  auto d_agg_kinds     = cudf::detail::make_device_uvector_async(agg_kinds, stream, mr);

  detail::rollup_row_equal row_equal{keys_dview,
                                     nullate,
                                     cudf::null_equality::EQUAL,
                                     num_levels,
                                     num_rolled,
                                     d_rolled_rank.data()};
  detail::rollup_row_hasher row_hash{
    keys_dview, nullate, num_levels, num_rolled, d_rolled_rank.data()};

  using probing_scheme_t = cuco::linear_probing<1, detail::rollup_row_hasher>;
  probing_scheme_t probing_scheme{row_hash};

  using rollup_global_set_t =
    cuco::static_set<cudf::size_type,
                     cuco::extent<int64_t>,
                     cuda::thread_scope_device,
                     detail::rollup_row_equal,
                     probing_scheme_t,
                     rmm::mr::polymorphic_allocator<char>,
                     cuco::storage<1>>;

  auto set = rollup_global_set_t{cuco::extent<int64_t>{static_cast<int64_t>(num_virtual)},
                                 cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                 cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
                                 row_equal,
                                 probing_scheme,
                                 cuco::thread_scope_device,
                                 cuco::storage<1>{},
                                 rmm::mr::polymorphic_allocator<char>{},
                                 stream.value()};

  bool const exclude_null_keys = null_handling_ == cudf::null_policy::EXCLUDE;

  auto set_ref_insert_find = set.ref(cuco::op::insert_and_find);
  constexpr int block_size = 256;
  cudf::size_type const num_sms =
    std::max(detail::device_multiprocessor_count(), cudf::size_type{1});
  cudf::size_type const rows_per_block =
    num_input_rows > 0 ? cudf::util::div_rounding_up_safe(num_input_rows, num_sms) : 0;
  int const grid_size = static_cast<int>(num_sms);
  if (num_input_rows > 0) {
    detail::rollup_aggregate_insert_find_kernel<<<grid_size, block_size, 0, stream.value()>>>(
      num_input_rows,
      rows_per_block,
      num_levels,
      exclude_null_keys,
      keys_dview,
      d_rolled_rank.data(),
      num_rolled,
      *d_input_values,
      *d_output_values,
      d_agg_kinds.data(),
      set_ref_insert_find);
    CUDF_CUDA_TRY(cudaPeekAtLastError());
  }

  rmm::device_uvector<cudf::size_type> unique_virtual_indices(
    static_cast<std::size_t>(num_virtual), stream, mr);
  auto const keys_end = set.retrieve_all(unique_virtual_indices.begin(), stream.value());
  unique_virtual_indices.resize(
    static_cast<std::size_t>(std::distance(unique_virtual_indices.begin(), keys_end)), stream);

  auto dense_agg_table = cudf::detail::gather(
    sparse_agg_results->view(),
    cudf::device_span<cudf::size_type const>{unique_virtual_indices.data(),
                                             unique_virtual_indices.size()},
    cudf::out_of_bounds_policy::DONT_CHECK,
    cudf::detail::negative_index_policy::NOT_ALLOWED,
    stream,
    mr);

  auto unique_keys_table = materialize_rollup_output_keys(
    keys_, unique_virtual_indices, num_levels, num_rolled, h_rank, stream, mr);

  auto gid_col = make_spark_grouping_id_column(unique_virtual_indices, num_levels, stream, mr);
  unique_keys_table = append_column_after_keys(
    std::move(unique_keys_table), std::move(gid_col), keys_.num_columns());

  cudf::detail::result_cache cache(requests.size());
  cudf::groupby::detail::hash::finalize_output(values, aggs, dense_agg_table, &cache, stream);

  auto [row_bitmask_storage, row_bitmask] =
    rollup_keys_row_bitmask(keys_, exclude_null_keys, stream, mr);

  if (has_compound_aggs) {
    for (auto const& request : requests) {
      auto const finalizer = cudf::groupby::detail::hash::hash_compound_agg_finalizer(
        request.values, &cache, row_bitmask, stream, mr);
      for (auto const& agg : request.aggregations) {
        cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
      }
    }
  }

  auto agg_results = cudf::groupby::detail::extract_results(requests, cache, stream, mr);
  return {std::move(unique_keys_table), std::move(agg_results)};
}

}  // namespace spark_rapids_jni
