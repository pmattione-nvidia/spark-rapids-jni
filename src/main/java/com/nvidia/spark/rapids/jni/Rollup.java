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

package com.nvidia.spark.rapids.jni;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.GroupByAggregation;
import ai.rapids.cudf.GroupByAggregationOnColumn;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;


/**
 * JNI entry for fused Spark ROLLUP partial aggregation ({@code spark_rapids_jni::rollup}).
 *
 * <p>Output column order matches fused expand+aggregate kernels (e.g. Ferdinand): {@code [key
 * columns..., spark_grouping_id (INT64), aggregate result columns...]}. Plain group-by without
 * rollup continues to use {@link Table.GroupByOperation} on the plugin side.
 *
 * <p>{@code ignoreNullKeys} maps to libcudf {@code null_policy} ({@code true} -> exclude rows
 * with null in active keys, {@code false} -> include). The Spark plugin passes {@code false},
 * matching {@code GroupByOptions.Builder#withIgnoreNullKeys(false)} on the non-fused path.
 *
 * <p>{@code rolledUpKeyIndicesAmongKeys} uses positions within {@code keyIndices} (0 .. {@code
 * keyIndices.length - 1}), not column indices in the input table.</p>
 */
public final class Rollup {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private Rollup() {}

  /**
   * Groups by {@code keyIndices}, applies ROLLUP on the key columns selected by {@code
   * rolledUpKeyIndicesAmongKeys}, and evaluates {@code aggregates} (same packing rules as
   * {@link Table.GroupByOperation#aggregate}).
   */
  public static Table aggregate(
      Table table,
      int[] keyIndices,
      int[] rolledUpKeyIndicesAmongKeys,
      boolean ignoreNullKeys,
      boolean keySorted,
      boolean[] keysDescending,
      boolean[] keysNullSmallest,
      GroupByAggregationOnColumn... aggregates) {
    if (keyIndices == null || keyIndices.length == 0) {
      throw new IllegalArgumentException("keyIndices must be non-null and non-empty");
    }
    if (aggregates == null || aggregates.length == 0) {
      throw new IllegalArgumentException("aggregates must be non-null and non-empty");
    }
    int[] rolled = rolledUpKeyIndicesAmongKeys == null ? new int[0] : rolledUpKeyIndicesAmongKeys;
    for (int r : rolled) {
      if (r < 0 || r >= keyIndices.length) {
        throw new IllegalArgumentException(
            "rolledUpKeyIndicesAmongKeys must reference key positions in [0, "
                + keyIndices.length
                + "): invalid index "
                + r);
      }
    }
    boolean[] kd = keysDescending == null ? new boolean[0] : keysDescending;
    boolean[] knf = keysNullSmallest == null ? new boolean[0] : keysNullSmallest;
    /** One column after data keys: {@code spark_grouping_id}. */
    final int gidExtra = 1;

    int keysLength = keyIndices.length;
    int[] aggColumnIndexes = new int[aggregates.length];
    long[] aggOperationInstances = new long[aggregates.length];
    try {
      for (int i = 0; i < aggregates.length; i++) {
        GroupByAggregationOnColumn agg = aggregates[i];
        aggColumnIndexes[i] = agg.getColumnIndex();
        aggOperationInstances[i] = agg.getWrapped().createNativeInstance();
      }

      try (Table out =
          new Table(
              nativeAggregate(
                  table.getNativeView(),
                  keyIndices,
                  rolled,
                  aggColumnIndexes,
                  aggOperationInstances,
                  ignoreNullKeys,
                  keySorted,
                  kd,
                  knf))) {
        ColumnVector[] finalCols = new ColumnVector[keysLength + gidExtra + aggregates.length];
        for (int k = 0; k < keysLength; k++) {
          finalCols[k] = out.getColumn(k);
        }
        finalCols[keysLength] = out.getColumn(keysLength);
        int aggBase = keysLength + gidExtra;
        for (int i = 0; i < aggregates.length; i++) {
          finalCols[aggBase + i] = out.getColumn(aggBase + i);
        }
        return new Table(finalCols);
      }
    } finally {
      GroupByAggregation.closeAggregations(aggOperationInstances);
    }
  }

  private static native long[] nativeAggregate(
      long tableView,
      int[] keyIndices,
      int[] rolledUpKeyIndicesAmongKeys,
      int[] aggColumnIndexes,
      long[] aggOperationInstances,
      boolean ignoreNullKeys,
      boolean keySorted,
      boolean[] keysDescending,
      boolean[] keysNullSmallest) throws CudfException;

}
