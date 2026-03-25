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

#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>

#include <cstdint>

namespace spark_rapids_jni::detail {

/**
 * @brief For each output key row (one per unique virtual index), mark BOOL8 valid where the key
 *        column is active for that row's grouping level; leave null where the column is rolled away.
 */
CUDF_KERNEL void rollup_active_key_column_kernel(cudf::mutable_column_device_view active_bool,
                                                 cudf::size_type const* unique_virtual,
                                                 cudf::size_type num_unique,
                                                 cudf::size_type rolled_rank_c,
                                                 cudf::size_type num_rolled,
                                                 cudf::size_type num_levels)
{
  auto const i = static_cast<cudf::size_type>(blockDim.x * blockIdx.x + threadIdx.x);
  if (i >= num_unique) { return; }

  auto const v     = unique_virtual[i];
  auto const level = v % num_levels;
  if (rollup_is_column_active(rolled_rank_c, num_rolled, level)) {
    active_bool.set_valid(i);
    active_bool.element<uint8_t>(i) = 1;
  }
}

}  // namespace spark_rapids_jni::detail
