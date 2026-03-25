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

#include "cudf_jni_apis.hpp"
#include "jni_utils.hpp"
#include "rollup.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <memory>
#include <vector>

namespace cudf::jni {
// Defined in TableJni.cpp (libcudfjni / spark_rapids_jni link).
std::vector<cudf::order> resolve_column_order(JNIEnv* env,
                                              jbooleanArray jkeys_sort_desc,
                                              int key_size);
std::vector<cudf::null_order> resolve_null_precedence(JNIEnv* env,
                                                      jbooleanArray jkeys_null_first,
                                                      int key_size);
}  // namespace cudf::jni

extern "C" {

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_Rollup_nativeAggregate(
  JNIEnv* env,
  jclass,
  jlong j_input_table,
  jintArray j_key_indices,
  jintArray j_rolled_up_among_keys,
  jintArray j_agg_column_indexes,
  jlongArray j_agg_instances,
  jboolean j_ignore_null_keys,
  jboolean j_key_sorted,
  jbooleanArray j_keys_sort_desc,
  jbooleanArray j_keys_null_first)
{
  JNI_NULL_CHECK(env, j_input_table, "input table is null", nullptr);
  JNI_NULL_CHECK(env, j_key_indices, "key indices are null", nullptr);
  JNI_NULL_CHECK(env, j_rolled_up_among_keys, "rolled up key indices are null", nullptr);
  JNI_NULL_CHECK(env, j_agg_column_indexes, "aggregate column indices are null", nullptr);
  JNI_NULL_CHECK(env, j_agg_instances, "aggregation instances are null", nullptr);

  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* n_input_table = reinterpret_cast<cudf::table_view*>(j_input_table);
    cudf::jni::native_jintArray n_keys(env, j_key_indices);
    cudf::jni::native_jintArray n_rolled(env, j_rolled_up_among_keys);
    cudf::jni::native_jintArray n_values(env, j_agg_column_indexes);
    cudf::jni::native_jpointerArray<cudf::aggregation> n_agg_instances(env, j_agg_instances);

    std::vector<cudf::column_view> n_keys_cols;
    n_keys_cols.reserve(n_keys.size());
    for (int i = 0; i < n_keys.size(); i++) {
      n_keys_cols.push_back(n_input_table->column(n_keys[i]));
    }
    cudf::table_view n_keys_table(n_keys_cols);

    spark_rapids_jni::rollup_spec spec;
    spec.rolled_up_key_column_indices.assign(n_rolled.begin(), n_rolled.end());

    auto column_order    = cudf::jni::resolve_column_order(env, j_keys_sort_desc, n_keys.size());
    auto null_precedence = cudf::jni::resolve_null_precedence(env, j_keys_null_first, n_keys.size());

    spark_rapids_jni::rollup roller(
      n_keys_table,
      std::move(spec),
      j_ignore_null_keys ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE,
      j_key_sorted ? cudf::sorted::YES : cudf::sorted::NO,
      column_order,
      null_precedence);

    std::vector<spark_rapids_jni::rollup_aggregation_request> requests;
    int previous_index = -1;
    for (int i = 0; i < n_values.size(); i++) {
      spark_rapids_jni::rollup_aggregation_request req;
      int col_index = n_values[i];

      auto* agg = dynamic_cast<cudf::groupby_aggregation*>(n_agg_instances[i]);
      JNI_ARG_CHECK(
        env, agg != nullptr, "aggregation is not an instance of groupby_aggregation", nullptr);
      std::unique_ptr<cudf::groupby_aggregation> cloned(
        dynamic_cast<cudf::groupby_aggregation*>(agg->clone().release()));

      if (col_index == previous_index) {
        requests.back().aggregations.push_back(std::move(cloned));
      } else {
        req.values = n_input_table->column(col_index);
        req.aggregations.push_back(std::move(cloned));
        requests.push_back(std::move(req));
      }
      previous_index = col_index;
    }

    auto result = roller.aggregate(
      cudf::host_span<spark_rapids_jni::rollup_aggregation_request const>(requests.data(),
                                                                            requests.size()));

    std::vector<std::unique_ptr<cudf::column>> result_columns;
    int agg_result_size = static_cast<int>(result.second.size());
    for (int agg_result_index = 0; agg_result_index < agg_result_size; agg_result_index++) {
      int col_agg_size = static_cast<int>(result.second[agg_result_index].results.size());
      for (int col_agg_index = 0; col_agg_index < col_agg_size; col_agg_index++) {
        result_columns.push_back(
          std::move(result.second[agg_result_index].results[col_agg_index]));
      }
    }
    return cudf::jni::convert_table_for_return(env, result.first, std::move(result_columns));
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"
