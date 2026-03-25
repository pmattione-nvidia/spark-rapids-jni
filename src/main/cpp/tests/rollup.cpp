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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using sum_type    = cudf::detail::target_type_t<int32_t, cudf::aggregation::SUM>;
using sum_i64_val = cudf::detail::target_type_t<int64_t, cudf::aggregation::SUM>;

struct RollupTest : public cudf::test::BaseFixture {};

namespace {

[[nodiscard]] std::pair<std::unique_ptr<cudf::table>, std::vector<spark_rapids_jni::rollup_aggregation_result>>
run_rollup_sum(cudf::table_view const& keys,
               cudf::column_view const& values,
               std::vector<cudf::size_type> rolled_indices)
{
  spark_rapids_jni::rollup_spec spec;
  spec.rolled_up_key_column_indices = std::move(rolled_indices);
  spark_rapids_jni::rollup roller(
    keys, spec, cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
  std::vector<spark_rapids_jni::rollup_aggregation_request> requests(1);
  requests[0].values = values;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  return roller.aggregate(requests, cudf::test::get_default_stream());
}

/**
 * Output row order from the hash path is not fixed; sort by spark_grouping_id, then keys (nulls
 * last), then sum — mirrors com.nvidia.spark.rapids.jni.RollupTest.
 */
[[nodiscard]] std::unique_ptr<cudf::table> sort_rollup_output_for_compare(
  cudf::table_view const& keys_and_gid, cudf::column_view const& sum_col)
{
  auto const nk = keys_and_gid.num_columns() - 1;  // last column is INT64 grouping id
  CUDF_EXPECTS(
    sum_col.size() == keys_and_gid.num_rows(), "rollup test: sum column row count must match keys");

  std::vector<cudf::column_view> sort_keys;
  sort_keys.reserve(static_cast<std::size_t>(nk + 2));
  sort_keys.push_back(keys_and_gid.column(nk));
  for (cudf::size_type c = 0; c < nk; ++c) {
    sort_keys.push_back(keys_and_gid.column(c));
  }
  sort_keys.push_back(sum_col);

  auto const num_sort = static_cast<cudf::size_type>(sort_keys.size());
  std::vector<cudf::order> orders(static_cast<std::size_t>(num_sort), cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_orders(static_cast<std::size_t>(num_sort),
                                            cudf::null_order::AFTER);

  auto gather_map = cudf::sorted_order(cudf::table_view(sort_keys), orders, null_orders);
  auto gathered_keys_gid = cudf::gather(keys_and_gid, *gather_map);
  auto gathered_sum      = cudf::gather(cudf::table_view{{sum_col}}, *gather_map);

  std::vector<std::unique_ptr<cudf::column>> cols = gathered_keys_gid->release();
  cols.push_back(std::move(gathered_sum->release().front()));
  return std::make_unique<cudf::table>(std::move(cols));
}

// -----------------------------------------------------------------------------
// Host reference for Spark ROLLUP + SUM (null_policy::INCLUDE on keys), matching
// sr_alfred integration_tests/.../fused_kernel_test.py scenarios.
// -----------------------------------------------------------------------------

[[nodiscard]] bool rollup_col_active(cudf::size_type rolled_rank,
                                     cudf::size_type num_rolled,
                                     cudf::size_type grouping_level)
{
  if (rolled_rank == num_rolled) { return true; }
  return rolled_rank < (num_rolled - grouping_level);
}

[[nodiscard]] int64_t rollup_spark_grouping_id(cudf::size_type grouping_level)
{
  return (1LL << static_cast<unsigned>(grouping_level)) - 1;
}

[[nodiscard]] std::vector<cudf::size_type> rollup_rolled_ranks(
  cudf::size_type num_key_cols, std::vector<cudf::size_type> const& rolled_indices)
{
  auto const num_rolled = static_cast<cudf::size_type>(rolled_indices.size());
  std::vector<cudf::size_type> ranks(static_cast<std::size_t>(num_key_cols), num_rolled);
  for (std::size_t i = 0; i < rolled_indices.size(); ++i) {
    ranks[static_cast<std::size_t>(rolled_indices[i])] = static_cast<cudf::size_type>(i);
  }
  return ranks;
}

// libcudf test column wrappers require a real ForwardIterator for validity; std::vector<bool>
// iterators are proxy iterators and produce wrong null masks.
[[nodiscard]] inline std::vector<uint8_t> rollup_validity_u8(std::vector<bool> const& mask)
{
  std::vector<uint8_t> out(mask.size());
  for (std::size_t i = 0; i < mask.size(); ++i) {
    out[i] = mask[i] ? uint8_t{1} : uint8_t{0};
  }
  return out;
}

template <typename SumT>
[[nodiscard]] bool less_for_sorted_rollup_compare(std::tuple<int64_t,
                                                                std::optional<int32_t>,
                                                                std::optional<int32_t>,
                                                                std::optional<int32_t>,
                                                                std::optional<SumT>> const& a,
                                                  std::tuple<int64_t,
                                                             std::optional<int32_t>,
                                                             std::optional<int32_t>,
                                                             std::optional<int32_t>,
                                                             std::optional<SumT>> const& b)
{
  // Order matches sort_rollup_output_for_compare: gid, k0, k1, k2, sum — ascending, nulls after
  // (non-null keys before null; libcudf null_order::AFTER).
  if (std::get<0>(a) != std::get<0>(b)) { return std::get<0>(a) < std::get<0>(b); }
  if (std::get<1>(a).has_value() != std::get<1>(b).has_value()) {
    return std::get<1>(a).has_value();
  }
  if (std::get<1>(a).has_value() && *std::get<1>(a) != *std::get<1>(b)) {
    return *std::get<1>(a) < *std::get<1>(b);
  }
  if (std::get<2>(a).has_value() != std::get<2>(b).has_value()) {
    return std::get<2>(a).has_value();
  }
  if (std::get<2>(a).has_value() && *std::get<2>(a) != *std::get<2>(b)) {
    return *std::get<2>(a) < *std::get<2>(b);
  }
  if (std::get<3>(a).has_value() != std::get<3>(b).has_value()) {
    return std::get<3>(a).has_value();
  }
  if (std::get<3>(a).has_value() && *std::get<3>(a) != *std::get<3>(b)) {
    return *std::get<3>(a) < *std::get<3>(b);
  }
  auto const& sa = std::get<4>(a);
  auto const& sb = std::get<4>(b);
  if (sa.has_value() != sb.has_value()) { return sa.has_value(); }
  if (sa.has_value() && *sa != *sb) { return *sa < *sb; }
  return false;
}

template <typename SumT>
[[nodiscard]] bool less_str2(std::tuple<int64_t,
                                       std::optional<std::string>,
                                       std::optional<std::string>,
                                       SumT> const& a,
                             std::tuple<int64_t,
                                        std::optional<std::string>,
                                        std::optional<std::string>,
                                        SumT> const& b)
{
  if (std::get<0>(a) != std::get<0>(b)) { return std::get<0>(a) < std::get<0>(b); }
  for (int c = 1; c <= 2; ++c) {
    auto const& xa = c == 1 ? std::get<1>(a) : std::get<2>(a);
    auto const& xb = c == 1 ? std::get<1>(b) : std::get<2>(b);
    if (xa.has_value() != xb.has_value()) { return xa.has_value(); }
    if (xa.has_value() && *xa != *xb) { return *xa < *xb; }
  }
  return std::get<3>(a) < std::get<3>(b);
}

template <typename SumT>
[[nodiscard]] bool less_i64_2(std::tuple<int64_t,
                                         std::optional<int64_t>,
                                         std::optional<int64_t>,
                                         SumT> const& a,
                              std::tuple<int64_t,
                                         std::optional<int64_t>,
                                         std::optional<int64_t>,
                                         SumT> const& b)
{
  if (std::get<0>(a) != std::get<0>(b)) { return std::get<0>(a) < std::get<0>(b); }
  for (int c = 1; c <= 2; ++c) {
    auto const& xa = c == 1 ? std::get<1>(a) : std::get<2>(a);
    auto const& xb = c == 1 ? std::get<1>(b) : std::get<2>(b);
    if (xa.has_value() != xb.has_value()) { return xa.has_value(); }
    if (xa.has_value() && *xa != *xb) { return *xa < *xb; }
  }
  return std::get<3>(a) < std::get<3>(b);
}

template <typename SumT>
void assert_sorted_rollup_matches_expected(
  std::unique_ptr<cudf::table> const& sorted_gpu,
  std::vector<std::tuple<int64_t,
                         std::optional<int32_t>,
                         std::optional<int32_t>,
                         std::optional<int32_t>,
                         std::optional<SumT>>> const& expected_rows,
  cudf::size_type num_key_cols)
{
  ASSERT_EQ(sorted_gpu->num_columns(), num_key_cols + 2);
  ASSERT_EQ(sorted_gpu->num_rows(), static_cast<cudf::size_type>(expected_rows.size()));

  std::vector<int32_t> ek0, ek1, ek2;
  std::vector<bool> vk0, vk1, vk2;
  std::vector<int64_t> egid;
  std::vector<SumT> esum;
  std::vector<bool> vsum;

  for (auto const& row : expected_rows) {
    egid.push_back(std::get<0>(row));
    auto push_opt = [&](std::optional<int32_t> const& o, std::vector<int32_t>& kv, std::vector<bool>& vv) {
      if (o) {
        kv.push_back(*o);
        vv.push_back(true);
      } else {
        kv.push_back(0);
        vv.push_back(false);
      }
    };
    if (num_key_cols >= 1) { push_opt(std::get<1>(row), ek0, vk0); }
    if (num_key_cols >= 2) { push_opt(std::get<2>(row), ek1, vk1); }
    if (num_key_cols >= 3) { push_opt(std::get<3>(row), ek2, vk2); }
    auto const& os = std::get<4>(row);
    if (os) {
      esum.push_back(*os);
      vsum.push_back(true);
    } else {
      esum.push_back(SumT{});
      vsum.push_back(false);
    }
  }

  auto make_k = [&](std::vector<int32_t> const& data, std::vector<bool> const& vv) {
    if (std::all_of(vv.begin(), vv.end(), [](bool v) { return v; })) {
      return cudf::test::fixed_width_column_wrapper<int32_t>(data.begin(), data.end());
    }
    auto const vu = rollup_validity_u8(vv);
    return cudf::test::fixed_width_column_wrapper<int32_t>(data.begin(), data.end(), vu.begin());
  };

  cudf::test::fixed_width_column_wrapper<int64_t> wgid(egid.begin(), egid.end());
  auto const wsum = std::all_of(vsum.begin(), vsum.end(), [](bool v) { return v; })
                      ? cudf::test::fixed_width_column_wrapper<SumT>(esum.begin(), esum.end())
                      : cudf::test::fixed_width_column_wrapper<SumT>(
                          esum.begin(), esum.end(), rollup_validity_u8(vsum).begin());

  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(num_key_cols), wgid);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(num_key_cols + 1), wsum);
  if (num_key_cols >= 1) { CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(0), make_k(ek0, vk0)); }
  if (num_key_cols >= 2) { CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(1), make_k(ek1, vk1)); }
  if (num_key_cols >= 3) { CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(2), make_k(ek2, vk2)); }
}

void assert_sorted_rollup_string2_matches(
  std::unique_ptr<cudf::table> const& sorted_gpu,
  std::vector<std::tuple<int64_t, std::optional<std::string>, std::optional<std::string>, sum_type>> const&
    expected_rows)
{
  ASSERT_EQ(sorted_gpu->num_columns(), 4);
  ASSERT_EQ(sorted_gpu->num_rows(), static_cast<cudf::size_type>(expected_rows.size()));

  std::vector<std::string> s0, s1;
  std::vector<bool> v0, v1;
  std::vector<int64_t> egid;
  std::vector<sum_type> esum;
  for (auto const& row : expected_rows) {
    egid.push_back(std::get<0>(row));
    auto const& a = std::get<1>(row);
    auto const& b = std::get<2>(row);
    if (a) {
      s0.push_back(*a);
      v0.push_back(true);
    } else {
      s0.emplace_back();
      v0.push_back(false);
    }
    if (b) {
      s1.push_back(*b);
      v1.push_back(true);
    } else {
      s1.emplace_back();
      v1.push_back(false);
    }
    esum.push_back(std::get<3>(row));
  }
  cudf::test::strings_column_wrapper wk0(s0.begin(), s0.end(), v0.begin());
  cudf::test::strings_column_wrapper wk1(s1.begin(), s1.end(), v1.begin());
  cudf::test::fixed_width_column_wrapper<int64_t> wgid(egid.begin(), egid.end());
  cudf::test::fixed_width_column_wrapper<sum_type> wsum(esum.begin(), esum.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(0), wk0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(1), wk1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(2), wgid);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(3), wsum);
}

void assert_sorted_rollup_i64_2_matches(
  std::unique_ptr<cudf::table> const& sorted_gpu,
  std::vector<std::tuple<int64_t, std::optional<int64_t>, std::optional<int64_t>, sum_i64_val>> const&
    expected_rows)
{
  ASSERT_EQ(sorted_gpu->num_columns(), 4);
  ASSERT_EQ(sorted_gpu->num_rows(), static_cast<cudf::size_type>(expected_rows.size()));
  std::vector<int64_t> k0, k1;
  std::vector<bool> vk0, vk1;
  std::vector<int64_t> egid;
  std::vector<sum_i64_val> esum;
  for (auto const& row : expected_rows) {
    egid.push_back(std::get<0>(row));
    auto push_opt = [&](std::optional<int64_t> const& o, std::vector<int64_t>& kv, std::vector<bool>& vv) {
      if (o) {
        kv.push_back(*o);
        vv.push_back(true);
      } else {
        kv.push_back(0);
        vv.push_back(false);
      }
    };
    push_opt(std::get<1>(row), k0, vk0);
    push_opt(std::get<2>(row), k1, vk1);
    esum.push_back(std::get<3>(row));
  }
  auto make_k = [&](std::vector<int64_t> const& data, std::vector<bool> const& vv) {
    if (std::all_of(vv.begin(), vv.end(), [](bool v) { return v; })) {
      return cudf::test::fixed_width_column_wrapper<int64_t>(data.begin(), data.end());
    }
    auto const vu = rollup_validity_u8(vv);
    return cudf::test::fixed_width_column_wrapper<int64_t>(data.begin(), data.end(), vu.begin());
  };
  auto const w0 = make_k(k0, vk0);
  auto const w1 = make_k(k1, vk1);
  cudf::test::fixed_width_column_wrapper<int64_t> wgid(egid.begin(), egid.end());
  cudf::test::fixed_width_column_wrapper<sum_i64_val> wsum(esum.begin(), esum.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(0), w0);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(1), w1);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(2), wgid);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted_gpu->get_column(3), wsum);
}

template <typename SumT>
void run_reference_rollup_int_keys(std::vector<int32_t> const& k0,
                                   std::vector<bool> const& m0,
                                   std::vector<int32_t> const& k1,
                                   std::vector<bool> const& m1,
                                   std::vector<int32_t> const& k2,
                                   std::vector<bool> const& m2,
                                   std::vector<SumT> const& vals,
                                   std::vector<bool> const& mv,
                                   std::vector<cudf::size_type> const& rolled_indices,
                                   bool coalesce_null_measures_to_zero,
                                   std::vector<std::tuple<int64_t,
                                                          std::optional<int32_t>,
                                                          std::optional<int32_t>,
                                                          std::optional<int32_t>,
                                                          std::optional<SumT>>>& out_sorted)
{
  auto const n = static_cast<cudf::size_type>(vals.size());
  CUDF_EXPECTS(static_cast<std::size_t>(n) == k0.size(), "ref: size");
  auto const num_keys = static_cast<int>(rolled_indices.size());
  CUDF_EXPECTS(num_keys >= 1 && num_keys <= 3, "ref: 1..3 rolled key columns");

  auto const ranks     = rollup_rolled_ranks(static_cast<cudf::size_type>(num_keys), rolled_indices);
  auto const num_rolled = static_cast<cudf::size_type>(rolled_indices.size());
  auto const nlev       = num_rolled + 1;

  using AccKey = std::tuple<int64_t, std::optional<int32_t>, std::optional<int32_t>, std::optional<int32_t>>;
  // Mirrors rollup_aggregate_insert_find_kernel: every (row, level) inserts into the set; SUM skips
  // null measures (element_aggregator) so the sparse slot stays invalid until a valid value arrives.
  std::map<AccKey, std::optional<SumT>> acc;

  auto proj = [&](int c, cudf::size_type row, cudf::size_type level) -> std::optional<int32_t> {
    if (not rollup_col_active(ranks[static_cast<std::size_t>(c)], num_rolled, level)) {
      return std::nullopt;
    }
    if (c == 0) {
      return m0[static_cast<std::size_t>(row)] ? std::optional<int32_t>{k0[row]} : std::nullopt;
    }
    if (c == 1) {
      return m1[static_cast<std::size_t>(row)] ? std::optional<int32_t>{k1[row]} : std::nullopt;
    }
    return m2[static_cast<std::size_t>(row)] ? std::optional<int32_t>{k2[row]} : std::nullopt;
  };

  for (cudf::size_type r = 0; r < n; ++r) {
    for (cudf::size_type level = 0; level < nlev; ++level) {
      AccKey key{rollup_spark_grouping_id(level), std::nullopt, std::nullopt, std::nullopt};
      if (num_keys >= 1) { std::get<1>(key) = proj(0, r, level); }
      if (num_keys >= 2) { std::get<2>(key) = proj(1, r, level); }
      if (num_keys >= 3) { std::get<3>(key) = proj(2, r, level); }

      auto& slot = acc[key];
      if (coalesce_null_measures_to_zero) {
        SumT const contrib = mv[static_cast<std::size_t>(r)] ? vals[r] : SumT{0};
        if (not slot.has_value()) { slot = SumT{0}; }
        *slot += contrib;
      } else if (mv[static_cast<std::size_t>(r)]) {
        SumT const contrib = vals[r];
        if (not slot.has_value()) {
          slot = contrib;
        } else {
          *slot += contrib;
        }
      }
    }
  }

  out_sorted.clear();
  out_sorted.reserve(acc.size());
  for (auto const& [k, s] : acc) {
    out_sorted.emplace_back(std::get<0>(k),
                            std::get<1>(k),
                            std::get<2>(k),
                            std::get<3>(k),
                            s);
  }
  std::sort(out_sorted.begin(), out_sorted.end(), less_for_sorted_rollup_compare<SumT>);
}

void run_reference_rollup_string2(std::vector<std::string> const& k0,
                                  std::vector<bool> const& m0,
                                  std::vector<std::string> const& k1,
                                  std::vector<bool> const& m1,
                                  std::vector<sum_type> const& vals,
                                  std::vector<bool> const& mv,
                                  std::vector<std::tuple<int64_t,
                                                         std::optional<std::string>,
                                                         std::optional<std::string>,
                                                         sum_type>>& out_sorted)
{
  auto const n = static_cast<cudf::size_type>(vals.size());
  auto const ranks = rollup_rolled_ranks(2, {0, 1});
  auto const num_rolled = cudf::size_type{2};
  auto const nlev       = num_rolled + 1;
  using AccKey          = std::tuple<int64_t, std::optional<std::string>, std::optional<std::string>>;
  std::map<AccKey, sum_type> acc;

  for (cudf::size_type r = 0; r < n; ++r) {
    if (not mv[static_cast<std::size_t>(r)]) { continue; }
    for (cudf::size_type level = 0; level < nlev; ++level) {
      std::optional<std::string> p0;
      if (rollup_col_active(ranks[0], num_rolled, level)) {
        p0 = m0[static_cast<std::size_t>(r)] ? std::optional<std::string>{k0[r]} : std::nullopt;
      }
      std::optional<std::string> p1;
      if (rollup_col_active(ranks[1], num_rolled, level)) {
        p1 = m1[static_cast<std::size_t>(r)] ? std::optional<std::string>{k1[r]} : std::nullopt;
      }
      AccKey key{rollup_spark_grouping_id(level), p0, p1};
      acc[key] += vals[r];
    }
  }
  out_sorted.clear();
  for (auto const& [k, s] : acc) {
    out_sorted.emplace_back(std::get<0>(k), std::get<1>(k), std::get<2>(k), s);
  }
  std::sort(out_sorted.begin(), out_sorted.end(), less_str2<sum_type>);
}

void run_reference_rollup_i64_2(std::vector<int64_t> const& k0,
                                std::vector<bool> const& m0,
                                std::vector<int64_t> const& k1,
                                std::vector<bool> const& m1,
                                std::vector<int64_t> const& vals,
                                std::vector<bool> const& mv,
                                std::vector<std::tuple<int64_t,
                                                       std::optional<int64_t>,
                                                       std::optional<int64_t>,
                                                       sum_i64_val>>& out_sorted)
{
  auto const n = static_cast<cudf::size_type>(vals.size());
  auto const ranks = rollup_rolled_ranks(2, {0, 1});
  auto const num_rolled = cudf::size_type{2};
  auto const nlev       = num_rolled + 1;
  using AccKey          = std::tuple<int64_t, std::optional<int64_t>, std::optional<int64_t>>;
  std::map<AccKey, sum_i64_val> acc;

  for (cudf::size_type r = 0; r < n; ++r) {
    if (not mv[static_cast<std::size_t>(r)]) { continue; }
    for (cudf::size_type level = 0; level < nlev; ++level) {
      std::optional<int64_t> p0;
      if (rollup_col_active(ranks[0], num_rolled, level)) {
        p0 = m0[static_cast<std::size_t>(r)] ? std::optional<int64_t>{k0[r]} : std::nullopt;
      }
      std::optional<int64_t> p1;
      if (rollup_col_active(ranks[1], num_rolled, level)) {
        p1 = m1[static_cast<std::size_t>(r)] ? std::optional<int64_t>{k1[r]} : std::nullopt;
      }
      AccKey key{rollup_spark_grouping_id(level), p0, p1};
      acc[key] += static_cast<sum_i64_val>(vals[r]);
    }
  }
  out_sorted.clear();
  for (auto const& [k, s] : acc) {
    out_sorted.emplace_back(std::get<0>(k), std::get<1>(k), std::get<2>(k), s);
  }
  std::sort(out_sorted.begin(), out_sorted.end(), less_i64_2<sum_i64_val>);
}

}  // namespace

TEST_F(RollupTest, OneIntKeySumSingleInputRow)
{
  cudf::test::fixed_width_column_wrapper<int32_t> k{7};
  cudf::test::fixed_width_column_wrapper<int32_t> v{100};
  cudf::table_view const keys{{k}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, v, {0});
  ASSERT_EQ(1, agg_out.size());
  ASSERT_EQ(1, agg_out.front().results.size());
  auto const sorted =
    sort_rollup_output_for_compare(out_keys_gid->view(), agg_out.front().results.front()->view());

  cudf::test::fixed_width_column_wrapper<int32_t> expect_k({7, 0}, {1, 0});
  cudf::test::fixed_width_column_wrapper<int64_t> expect_gid{0, 1};
  cudf::test::fixed_width_column_wrapper<sum_type> expect_sum{100, 100};

  EXPECT_EQ(2, sorted->num_rows());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(0), expect_k);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(1), expect_gid);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(2), expect_sum);
}

TEST_F(RollupTest, OneIntKeySumTwoDistinctKeys)
{
  cudf::test::fixed_width_column_wrapper<int32_t> k{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> v{10, 20};
  cudf::table_view const keys{{k}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, v, {0});
  auto const sorted =
    sort_rollup_output_for_compare(out_keys_gid->view(), agg_out.front().results.front()->view());

  cudf::test::fixed_width_column_wrapper<int32_t> expect_k({1, 2, 0}, {1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int64_t> expect_gid{0, 0, 1};
  cudf::test::fixed_width_column_wrapper<sum_type> expect_sum{10, 20, 30};

  EXPECT_EQ(3, sorted->num_rows());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(0), expect_k);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(1), expect_gid);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(2), expect_sum);
}

TEST_F(RollupTest, OneIntKeySumDuplicateKeyAggregatesWithinFinestLevel)
{
  cudf::test::fixed_width_column_wrapper<int32_t> k{1, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> v{10, 20};
  cudf::table_view const keys{{k}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, v, {0});
  auto const sorted =
    sort_rollup_output_for_compare(out_keys_gid->view(), agg_out.front().results.front()->view());

  cudf::test::fixed_width_column_wrapper<int32_t> expect_k({1, 0}, {1, 0});
  cudf::test::fixed_width_column_wrapper<int64_t> expect_gid{0, 1};
  cudf::test::fixed_width_column_wrapper<sum_type> expect_sum{30, 30};

  EXPECT_EQ(2, sorted->num_rows());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(0), expect_k);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(1), expect_gid);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(2), expect_sum);
}

TEST_F(RollupTest, TwoIntKeysSumRollupBoth)
{
  cudf::test::fixed_width_column_wrapper<int32_t> k0{1, 1};
  cudf::test::fixed_width_column_wrapper<int32_t> k1{10, 20};
  cudf::test::fixed_width_column_wrapper<int32_t> v{100, 200};
  cudf::table_view const keys{{k0, k1}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, v, {0, 1});
  auto const sorted =
    sort_rollup_output_for_compare(out_keys_gid->view(), agg_out.front().results.front()->view());

  cudf::test::fixed_width_column_wrapper<int32_t> expect_k0({1, 1, 1, 0}, {1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<int32_t> expect_k1({10, 20, 0, 0}, {1, 1, 0, 0});
  cudf::test::fixed_width_column_wrapper<int64_t> expect_gid{0, 0, 1, 3};
  cudf::test::fixed_width_column_wrapper<sum_type> expect_sum{100, 200, 300, 300};

  EXPECT_EQ(4, sorted->num_rows());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(0), expect_k0);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(1), expect_k1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(2), expect_gid);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted->get_column(3), expect_sum);
}

TEST_F(RollupTest, EmptyRolledListOrdinaryGroupByWithGroupingId)
{
  cudf::test::fixed_width_column_wrapper<int32_t> k{1, 1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> v{10, 20, 30};
  cudf::table_view const keys{{k}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, v, {});
  auto const sorted =
    sort_rollup_output_for_compare(out_keys_gid->view(), agg_out.front().results.front()->view());

  cudf::test::fixed_width_column_wrapper<int32_t> expect_k{1, 2};
  cudf::test::fixed_width_column_wrapper<int64_t> expect_gid{0, 0};
  cudf::test::fixed_width_column_wrapper<sum_type> expect_sum{30, 30};

  EXPECT_EQ(2, sorted->num_rows());
  // Output may use a nullable column with no nulls; wrappers without a validity vector are not
  // nullable — EQUIVALENT compares data + logical nulls, not null-mask representation.
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted->get_column(0), expect_k);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted->get_column(1), expect_gid);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(sorted->get_column(2), expect_sum);
}

TEST_F(RollupTest, SingleRolledKeyProducesGrandTotalRowAndGroupingIdMultiset)
{
  cudf::test::fixed_width_column_wrapper<int32_t> k{1, 2};
  cudf::test::fixed_width_column_wrapper<int32_t> v{10, 20};
  cudf::table_view const keys{{k}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, v, {0});

  EXPECT_EQ(3, out_keys_gid->num_rows());
  EXPECT_EQ(2, out_keys_gid->num_columns());
  EXPECT_EQ(1, out_keys_gid->get_column(0).null_count());

  auto const gid_view = out_keys_gid->get_column(1).view();
  std::vector<cudf::column_view> const gid_table_cols{gid_view};
  cudf::table_view const gid_table{gid_table_cols};
  auto const gid_sort_map =
    cudf::sorted_order(gid_table, {cudf::order::ASCENDING}, {cudf::null_order::AFTER});
  auto const sorted_gid_tbl = cudf::gather(gid_table, *gid_sort_map);
  cudf::test::fixed_width_column_wrapper<int64_t> expect_gid_multiset{0, 0, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(sorted_gid_tbl->view().column(0), expect_gid_multiset);

  auto const sum_view = agg_out.front().results.front()->view();
  auto const sum_total =
    cudf::reduce(sum_view,
                 *cudf::make_sum_aggregation<cudf::reduce_aggregation>(),
                 cudf::data_type{cudf::type_id::INT64},
                 cudf::test::get_default_stream());
  auto const* ns = dynamic_cast<cudf::numeric_scalar<int64_t> const*>(sum_total.get());
  ASSERT_NE(nullptr, ns);
  EXPECT_EQ(60, ns->value());
}

// --- Fused-kernel style (cf. sr_alfred integration_tests/.../fused_kernel_test.py) ---

TEST_F(RollupTest, FusedKernelStyle_RepeatSeqTwoIntKeysSum5000Rows)
{
  constexpr int n = 5000;
  std::vector<int32_t> k0(static_cast<std::size_t>(n));
  std::vector<int32_t> k1(static_cast<std::size_t>(n));
  std::vector<sum_type> vals(static_cast<std::size_t>(n));
  std::vector<bool> const all_true(static_cast<std::size_t>(n), true);
  for (int i = 0; i < n; ++i) {
    k0[static_cast<std::size_t>(i)] = 1 + (i % 10);
    k1[static_cast<std::size_t>(i)] = 1 + (i % 20);
    vals[static_cast<std::size_t>(i)] = static_cast<sum_type>(1 + (i % 1000));
  }
  std::vector<int32_t> k2_pad(static_cast<std::size_t>(n));
  std::vector<std::tuple<int64_t, std::optional<int32_t>, std::optional<int32_t>, std::optional<int32_t>, std::optional<sum_type>>>
    expected;
  run_reference_rollup_int_keys(k0,
                                all_true,
                                k1,
                                all_true,
                                k2_pad,
                                all_true,
                                vals,
                                all_true,
                                {0, 1},
                                false,
                                expected);

  cudf::test::fixed_width_column_wrapper<int32_t> w0(k0.begin(), k0.end());
  cudf::test::fixed_width_column_wrapper<int32_t> w1(k1.begin(), k1.end());
  cudf::test::fixed_width_column_wrapper<sum_type> wv(vals.begin(), vals.end());
  cudf::table_view const keys{{w0, w1}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, wv, {0, 1});
  auto const sorted = sort_rollup_output_for_compare(out_keys_gid->view(),
                                                      agg_out.front().results.front()->view());
  assert_sorted_rollup_matches_expected(sorted, expected, 2);
}

TEST_F(RollupTest, FusedKernelStyle_ThreeIntKeysRolledSum5000Rows)
{
  constexpr int n = 5000;
  std::vector<int32_t> k0(static_cast<std::size_t>(n));
  std::vector<int32_t> k1(static_cast<std::size_t>(n));
  std::vector<int32_t> k2(static_cast<std::size_t>(n));
  std::vector<sum_type> vals(static_cast<std::size_t>(n));
  std::vector<bool> const all_true(static_cast<std::size_t>(n), true);
  for (int i = 0; i < n; ++i) {
    k0[static_cast<std::size_t>(i)] = 1 + (i % 10);
    k1[static_cast<std::size_t>(i)] = 1 + (i % 20);
    k2[static_cast<std::size_t>(i)] = 1 + (i % 5);
    vals[static_cast<std::size_t>(i)] = static_cast<sum_type>(1 + (i % 1000));
  }
  std::vector<std::tuple<int64_t, std::optional<int32_t>, std::optional<int32_t>, std::optional<int32_t>, std::optional<sum_type>>>
    expected;
  run_reference_rollup_int_keys(
    k0, all_true, k1, all_true, k2, all_true, vals, all_true, {0, 1, 2}, false, expected);

  cudf::test::fixed_width_column_wrapper<int32_t> w0(k0.begin(), k0.end());
  cudf::test::fixed_width_column_wrapper<int32_t> w1(k1.begin(), k1.end());
  cudf::test::fixed_width_column_wrapper<int32_t> w2(k2.begin(), k2.end());
  cudf::test::fixed_width_column_wrapper<sum_type> wv(vals.begin(), vals.end());
  cudf::table_view const keys{{w0, w1, w2}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, wv, {0, 1, 2});
  auto const sorted = sort_rollup_output_for_compare(out_keys_gid->view(),
                                                      agg_out.front().results.front()->view());
  assert_sorted_rollup_matches_expected(sorted, expected, 3);
}

TEST_F(RollupTest, FusedKernelStyle_SumProductPrecomputedColumn5000Rows)
{
  constexpr int n = 5000;
  std::vector<int32_t> k0(static_cast<std::size_t>(n));
  std::vector<int32_t> k1(static_cast<std::size_t>(n));
  std::vector<sum_type> prod(static_cast<std::size_t>(n));
  std::vector<bool> const all_true(static_cast<std::size_t>(n), true);
  for (int i = 0; i < n; ++i) {
    auto const m1 = 1 + (i % 1000);
    auto const mm = static_cast<int32_t>(1 + (i % 100));
    k0[static_cast<std::size_t>(i)]   = 1 + (i % 10);
    k1[static_cast<std::size_t>(i)]   = 1 + (i % 20);
    prod[static_cast<std::size_t>(i)] = static_cast<sum_type>(m1 * mm);
  }
  std::vector<int32_t> k2_pad(static_cast<std::size_t>(n));
  std::vector<std::tuple<int64_t, std::optional<int32_t>, std::optional<int32_t>, std::optional<int32_t>, std::optional<sum_type>>>
    expected;
  run_reference_rollup_int_keys(k0,
                                all_true,
                                k1,
                                all_true,
                                k2_pad,
                                all_true,
                                prod,
                                all_true,
                                {0, 1},
                                false,
                                expected);

  cudf::test::fixed_width_column_wrapper<int32_t> w0(k0.begin(), k0.end());
  cudf::test::fixed_width_column_wrapper<int32_t> w1(k1.begin(), k1.end());
  cudf::test::fixed_width_column_wrapper<sum_type> wp(prod.begin(), prod.end());
  cudf::table_view const keys{{w0, w1}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, wp, {0, 1});
  auto const sorted = sort_rollup_output_for_compare(out_keys_gid->view(),
                                                      agg_out.front().results.front()->view());
  assert_sorted_rollup_matches_expected(sorted, expected, 2);
}

TEST_F(RollupTest, FusedKernelStyle_NullableKeyColumnsPartialNulls)
{
  constexpr int n = 5000;
  std::vector<int32_t> k0(static_cast<std::size_t>(n));
  std::vector<int32_t> k1(static_cast<std::size_t>(n));
  std::vector<bool> m0(static_cast<std::size_t>(n), true);
  std::vector<bool> m1(static_cast<std::size_t>(n), true);
  std::vector<sum_type> vals(static_cast<std::size_t>(n));
  std::vector<bool> const mv(static_cast<std::size_t>(n), true);
  for (int i = 0; i < n; ++i) {
    k0[static_cast<std::size_t>(i)] = 1 + (i % 15);
    k1[static_cast<std::size_t>(i)] = 1 + (i % 15);
    m0[static_cast<std::size_t>(i)] = (i % 5) != 0;
    m1[static_cast<std::size_t>(i)] = (i % 7) != 0;
    vals[static_cast<std::size_t>(i)] = static_cast<sum_type>(1 + (i % 500));
  }
  std::vector<int32_t> k2_pad(static_cast<std::size_t>(n));
  std::vector<bool> m2_pad(static_cast<std::size_t>(n), true);
  std::vector<std::tuple<int64_t, std::optional<int32_t>, std::optional<int32_t>, std::optional<int32_t>, std::optional<sum_type>>>
    expected;
  run_reference_rollup_int_keys(
    k0, m0, k1, m1, k2_pad, m2_pad, vals, mv, {0, 1}, false, expected);

  auto const v0 = rollup_validity_u8(m0);
  auto const v1 = rollup_validity_u8(m1);
  auto const w0 = std::all_of(m0.begin(), m0.end(), [](bool x) { return x; })
                    ? cudf::test::fixed_width_column_wrapper<int32_t>(k0.begin(), k0.end())
                    : cudf::test::fixed_width_column_wrapper<int32_t>(k0.begin(), k0.end(), v0.begin());
  auto const w1 = std::all_of(m1.begin(), m1.end(), [](bool x) { return x; })
                    ? cudf::test::fixed_width_column_wrapper<int32_t>(k1.begin(), k1.end())
                    : cudf::test::fixed_width_column_wrapper<int32_t>(k1.begin(), k1.end(), v1.begin());
  cudf::test::fixed_width_column_wrapper<sum_type> wv(vals.begin(), vals.end());
  cudf::table_view const keys{{w0, w1}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, wv, {0, 1});
  auto const sorted = sort_rollup_output_for_compare(out_keys_gid->view(),
                                                      agg_out.front().results.front()->view());
  assert_sorted_rollup_matches_expected(sorted, expected, 2);
}

TEST_F(RollupTest, FusedKernelStyle_PartialNullMeasuresSumSkipsNulls)
{
  // Like Spark SUM(measure) with nullable measure: null inputs do not add (cf. fused_kernel
  // test_rollup_sum with nullable value gens — not the coalesce variant).
  constexpr int n = 5000;
  std::vector<int32_t> k0(static_cast<std::size_t>(n));
  std::vector<int32_t> k1(static_cast<std::size_t>(n));
  std::vector<sum_type> vals(static_cast<std::size_t>(n));
  std::vector<bool> mv(static_cast<std::size_t>(n));
  std::vector<bool> const all_true(static_cast<std::size_t>(n), true);
  for (int i = 0; i < n; ++i) {
    k0[static_cast<std::size_t>(i)] = 1 + (i % 10);
    k1[static_cast<std::size_t>(i)] = 1 + (i % 20);
    vals[static_cast<std::size_t>(i)] = static_cast<sum_type>(1 + (i % 1000));
    mv[static_cast<std::size_t>(i)]   = (i % 4) != 0;
  }
  std::vector<int32_t> k2_pad(static_cast<std::size_t>(n));
  std::vector<std::tuple<int64_t, std::optional<int32_t>, std::optional<int32_t>, std::optional<int32_t>, std::optional<sum_type>>>
    expected;
  run_reference_rollup_int_keys(k0,
                                all_true,
                                k1,
                                all_true,
                                k2_pad,
                                all_true,
                                vals,
                                mv,
                                {0, 1},
                                false,
                                expected);

  auto const vv = rollup_validity_u8(mv);
  auto const wv = std::all_of(mv.begin(), mv.end(), [](bool x) { return x; })
                    ? cudf::test::fixed_width_column_wrapper<sum_type>(vals.begin(), vals.end())
                    : cudf::test::fixed_width_column_wrapper<sum_type>(vals.begin(), vals.end(), vv.begin());
  cudf::test::fixed_width_column_wrapper<int32_t> w0(k0.begin(), k0.end());
  cudf::test::fixed_width_column_wrapper<int32_t> w1(k1.begin(), k1.end());
  cudf::table_view const keys{{w0, w1}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, wv, {0, 1});
  auto const sorted = sort_rollup_output_for_compare(out_keys_gid->view(),
                                                      agg_out.front().results.front()->view());
  assert_sorted_rollup_matches_expected(sorted, expected, 2);
}

TEST_F(RollupTest, FusedKernelStyle_Int64KeysTwoRolledSum5000Rows)
{
  constexpr int n = 5000;
  std::vector<int64_t> k0(static_cast<std::size_t>(n));
  std::vector<int64_t> k1(static_cast<std::size_t>(n));
  std::vector<int64_t> vals(static_cast<std::size_t>(n));
  std::vector<bool> const all_true(static_cast<std::size_t>(n), true);
  for (int i = 0; i < n; ++i) {
    k0[static_cast<std::size_t>(i)] = 1 + (i % 100);
    k1[static_cast<std::size_t>(i)] = 1 + (i % 100);
    vals[static_cast<std::size_t>(i)] = 1 + (i % 100000);
  }
  std::vector<std::tuple<int64_t, std::optional<int64_t>, std::optional<int64_t>, sum_i64_val>> expected;
  run_reference_rollup_i64_2(k0, all_true, k1, all_true, vals, all_true, expected);

  cudf::test::fixed_width_column_wrapper<int64_t> w0(k0.begin(), k0.end());
  cudf::test::fixed_width_column_wrapper<int64_t> w1(k1.begin(), k1.end());
  cudf::test::fixed_width_column_wrapper<int64_t> wv(vals.begin(), vals.end());
  cudf::table_view const keys{{w0, w1}};
  spark_rapids_jni::rollup_spec spec;
  spec.rolled_up_key_column_indices = {0, 1};
  spark_rapids_jni::rollup roller(
    keys, spec, cudf::null_policy::INCLUDE, cudf::sorted::NO, {}, {});
  std::vector<spark_rapids_jni::rollup_aggregation_request> requests(1);
  requests[0].values = wv;
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  auto [out_keys_gid, agg_out] = roller.aggregate(requests, cudf::test::get_default_stream());
  auto const sorted = sort_rollup_output_for_compare(out_keys_gid->view(),
                                                      agg_out.front().results.front()->view());
  assert_sorted_rollup_i64_2_matches(sorted, expected);
}

TEST_F(RollupTest, FusedKernelStyle_StringKeysTwoRolledSum2000Rows)
{
  constexpr int n = 2000;
  std::vector<std::string> k0;
  std::vector<std::string> k1;
  k0.reserve(static_cast<std::size_t>(n));
  k1.reserve(static_cast<std::size_t>(n));
  std::vector<sum_type> vals(static_cast<std::size_t>(n));
  std::vector<bool> const all_true(static_cast<std::size_t>(n), true);
  for (int i = 0; i < n; ++i) {
    k0.push_back(std::string("d1_") + static_cast<char>('0' + (i % 10)));
    k1.push_back(std::string("d2_") + static_cast<char>('0' + (i % 20)));
    vals[static_cast<std::size_t>(i)] = static_cast<sum_type>(1 + (i % 1000));
  }

  std::vector<std::tuple<int64_t, std::optional<std::string>, std::optional<std::string>, sum_type>> expected;
  run_reference_rollup_string2(k0, all_true, k1, all_true, vals, all_true, expected);

  cudf::test::strings_column_wrapper w0(k0.begin(), k0.end());
  cudf::test::strings_column_wrapper w1(k1.begin(), k1.end());
  cudf::test::fixed_width_column_wrapper<sum_type> wv(vals.begin(), vals.end());
  cudf::table_view const keys{{w0, w1}};
  auto [out_keys_gid, agg_out] = run_rollup_sum(keys, wv, {0, 1});
  auto const sorted = sort_rollup_output_for_compare(out_keys_gid->view(),
                                                      agg_out.front().results.front()->view());
  assert_sorted_rollup_string2_matches(sorted, expected);
}
