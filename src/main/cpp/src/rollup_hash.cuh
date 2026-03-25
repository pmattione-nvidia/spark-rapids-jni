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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace spark_rapids_jni::detail {

/**
 * @brief Virtual row index layout aligned with fused expand/rollup kernels:
 *        `virtual_index = input_row_index * num_levels + grouping_level`.
 *
 * `num_levels` is `rolled_key_count + 1`. `grouping_level` 0 is the finest grouping (all rolled
 * keys participate); `grouping_level == rolled_key_count` is coarsest (only fixed keys participate).
 */
[[nodiscard]] inline cudf::size_type rollup_num_levels(cudf::size_type num_rolled_keys)
{
  return num_rolled_keys + 1;
}

[[nodiscard]] inline cudf::size_type rollup_virtual_row_count(cudf::size_type input_rows,
                                                              cudf::size_type num_levels)
{
  return input_rows * num_levels;
}

[[nodiscard]] __device__ inline cudf::size_type rollup_virtual_to_input_row(cudf::size_type virtual_ix,
                                                                            cudf::size_type num_levels)
{
  return virtual_ix / num_levels;
}

[[nodiscard]] __device__ inline cudf::size_type rollup_virtual_to_grouping_level(
  cudf::size_type virtual_ix, cudf::size_type num_levels)
{
  return virtual_ix % num_levels;
}

/**
 * `rolled_rank[i]` is the position of key column `i` in the rollup suffix (0 .. num_rolled-1), or
 * `num_rolled` when column `i` is a fixed (non-rolled) grouping key.
 */
[[nodiscard]] __device__ inline bool rollup_is_column_active(cudf::size_type rolled_rank,
                                                             cudf::size_type num_rolled,
                                                             cudf::size_type grouping_level)
{
  if (rolled_rank == num_rolled) { return true; }
  return rolled_rank < (num_rolled - grouping_level);
}

/**
 * @brief For `null_policy::EXCLUDE`, skip a virtual row when any *active* key column is null.
 */
[[nodiscard]] __device__ inline bool rollup_skip_virtual_row_for_null_exclude(
  cudf::table_device_view table,
  cudf::size_type const* rolled_rank,
  cudf::size_type num_rolled,
  cudf::size_type num_levels,
  cudf::size_type virtual_ix)
{
  auto const row            = rollup_virtual_to_input_row(virtual_ix, num_levels);
  auto const grouping_level = rollup_virtual_to_grouping_level(virtual_ix, num_levels);
  for (cudf::size_type col = 0; col < table.num_columns(); ++col) {
    if (not rollup_is_column_active(rolled_rank[col], num_rolled, grouping_level)) { continue; }
    if (table.column(col).is_null(row)) { return true; }
  }
  return false;
}

namespace {

using nan_eq = cudf::detail::row::equality::nan_equal_physical_equality_comparator;

class rollup_element_hasher_adapter {
  using result_type = cuda::std::invoke_result_t<cudf::hashing::detail::default_hash<int32_t>,
                                                 int32_t>;
  static constexpr result_type NULL_HASH     = cuda::std::numeric_limits<result_type>::max();
  static constexpr result_type NON_NULL_HASH = 0;

  cudf::detail::row::hash::element_hasher<cudf::hashing::detail::default_hash,
                                          cudf::nullate::DYNAMIC> _element_hasher;
  cudf::nullate::DYNAMIC _check_nulls;

 public:
  __device__ rollup_element_hasher_adapter(cudf::nullate::DYNAMIC check_nulls, result_type seed)
    : _element_hasher(check_nulls, seed), _check_nulls(check_nulls)
  {
  }

  template <typename T>
  __device__ result_type operator()(cudf::column_device_view const& col,
                                    cudf::size_type row_index) const noexcept
    requires(not cudf::is_nested<T>() and not cudf::is_dictionary<T>())
  {
    return _element_hasher.template operator()<T>(col, row_index);
  }

  template <typename T>
  __device__ result_type operator()(cudf::column_device_view const& col,
                                    cudf::size_type row_index) const noexcept
    requires(cudf::is_dictionary<T>())
  {
    if (_check_nulls && col.is_null(row_index)) { return NULL_HASH; }

    auto const keys = col.child(cudf::dictionary_column_view::keys_column_index);
    return cudf::type_dispatcher<cudf::dispatch_storage_type>(
      keys.type(),
      _element_hasher,
      keys,
      static_cast<cudf::size_type>(col.element<cudf::dictionary32>(row_index)));
  }

  template <typename T>
  __device__ result_type operator()(cudf::column_device_view const&,
                                    cudf::size_type) const noexcept
    requires(cudf::is_nested<T>())
  {
    CUDF_UNREACHABLE("rollup hash: nested key columns are not supported");
  }
};

struct rollup_dict_keys_equal {
  cudf::column_device_view col;
  nan_eq comparator{};

  template <typename KeyType>
  __device__ bool operator()(cudf::size_type lhs_element_index,
                             cudf::size_type rhs_element_index) const noexcept
    requires(cudf::is_equality_comparable<KeyType, KeyType>())
  {
    auto const lidx = col.element<cudf::dictionary32>(lhs_element_index).value();
    auto const ridx = col.element<cudf::dictionary32>(rhs_element_index).value();
    auto const keys = col.child(cudf::dictionary_column_view::keys_column_index);
    return comparator(keys.element<KeyType>(lidx), keys.element<KeyType>(ridx));
  }

  template <typename KeyType, typename... Args>
  __device__ bool operator()(Args...) const noexcept
    requires(not cudf::is_equality_comparable<KeyType, KeyType>())
  {
    CUDF_UNREACHABLE("rollup equality: dictionary key types are not comparable");
  }
};

struct rollup_column_equal {
  cudf::nullate::DYNAMIC check_nulls;
  cudf::null_equality nulls_are_equal;
  cudf::column_device_view col;
  cudf::size_type r1;
  cudf::size_type r2;

  template <typename Element>
  __device__ bool operator()() const noexcept
    requires(cudf::is_equality_comparable<Element, Element>() and
             not cudf::is_dictionary<Element>() and not cudf::is_nested<Element>())
  {
    nan_eq comp{};
    if (check_nulls) {
      bool const lhs_is_null{col.is_null(r1)};
      bool const rhs_is_null{col.is_null(r2)};
      if (lhs_is_null and rhs_is_null) {
        return nulls_are_equal == cudf::null_equality::EQUAL;
      }
      if (lhs_is_null != rhs_is_null) { return false; }
    }
    return comp(col.element<Element>(r1), col.element<Element>(r2));
  }

  template <typename Element>
  __device__ bool operator()() const noexcept requires(cudf::is_dictionary<Element>())
  {
    if (check_nulls) {
      bool const lhs_is_null{col.is_null(r1)};
      bool const rhs_is_null{col.is_null(r2)};
      if (lhs_is_null and rhs_is_null) {
        return nulls_are_equal == cudf::null_equality::EQUAL;
      }
      if (lhs_is_null != rhs_is_null) { return false; }
    }

    return cudf::type_dispatcher<cudf::detail::dispatch_void_if_nested>(
      col.child(cudf::dictionary_column_view::keys_column_index).type(),
      rollup_dict_keys_equal{col},
      r1,
      r2);
  }

  template <typename Element>
  __device__ bool operator()() const noexcept requires(cudf::is_nested<Element>())
  {
    CUDF_UNREACHABLE("rollup equality: nested key columns are not supported");
  }

  template <typename Element, typename... Args>
  __device__ bool operator()(Args...) const noexcept
    requires(not cudf::is_equality_comparable<Element, Element>())
  {
    CUDF_UNREACHABLE("rollup equality: type is not equality comparable");
  }
};

}  // namespace

/**
 * @brief Probing hash for `cuco::static_set`: hash virtual row index using only active key columns.
 */
struct rollup_row_hasher {
  cudf::table_device_view table;
  cudf::nullate::DYNAMIC check_nullness;
  cudf::size_type num_levels{};
  cudf::size_type num_rolled{};
  cudf::size_type const* rolled_rank{};

  using result_type = cuda::std::invoke_result_t<cudf::hashing::detail::default_hash<int32_t>,
                                                 int32_t>;

  __device__ result_type operator()(cudf::size_type const virtual_ix) const noexcept
  {
    auto const row            = rollup_virtual_to_input_row(virtual_ix, num_levels);
    auto const grouping_level = rollup_virtual_to_grouping_level(virtual_ix, num_levels);

    result_type h = cudf::DEFAULT_HASH_SEED;
    h             = cudf::hashing::detail::hash_combine(h, static_cast<result_type>(grouping_level));

    rollup_element_hasher_adapter adapter{check_nullness, cudf::DEFAULT_HASH_SEED};

    for (cudf::size_type col = 0; col < table.num_columns(); ++col) {
      if (not rollup_is_column_active(rolled_rank[col], num_rolled, grouping_level)) { continue; }

      auto const& column = table.column(col);
      result_type col_hash =
        cudf::type_dispatcher<cudf::dispatch_storage_type>(column.type(), adapter, column, row);
      h = cudf::hashing::detail::hash_combine(h, col_hash);
    }
    return h;
  }
};

/**
 * @brief Equality of virtual indices for `cuco::static_set`: same grouping level and equal active
 *        keys.
 */
struct rollup_row_equal {
  cudf::table_device_view table;
  cudf::nullate::DYNAMIC check_nullness;
  cudf::null_equality null_keys_are_equal;
  cudf::size_type num_levels{};
  cudf::size_type num_rolled{};
  cudf::size_type const* rolled_rank{};

  __device__ bool operator()(cudf::size_type const v1, cudf::size_type const v2) const noexcept
  {
    auto const r1     = rollup_virtual_to_input_row(v1, num_levels);
    auto const r2     = rollup_virtual_to_input_row(v2, num_levels);
    auto const level1 = rollup_virtual_to_grouping_level(v1, num_levels);
    auto const level2 = rollup_virtual_to_grouping_level(v2, num_levels);
    if (level1 != level2) { return false; }

    for (cudf::size_type col = 0; col < table.num_columns(); ++col) {
      if (not rollup_is_column_active(rolled_rank[col], num_rolled, level1)) { continue; }

      auto const& c   = table.column(col);
      bool const c_eq = cudf::type_dispatcher<cudf::dispatch_storage_type>(
        c.type(),
        rollup_column_equal{check_nullness, null_keys_are_equal, c, r1, r2});
      if (not c_eq) { return false; }
    }
    return true;
  }
};

}  // namespace spark_rapids_jni::detail
