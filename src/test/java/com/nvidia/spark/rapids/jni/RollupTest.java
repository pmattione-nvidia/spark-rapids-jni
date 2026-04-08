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
import ai.rapids.cudf.DType;
import ai.rapids.cudf.GroupByAggregation;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class RollupTest {

  /**
   * Simplest case: one int grouping column (fully rolled), one int measure column, {@code SUM}.
   * One input row → detail row + grand total (same sum).
   */
  @Test
  void rollupOneIntKey_sum_singleInputRow() {
    try (Table t = new Table(ColumnVector.fromInts(7), ColumnVector.fromInts(100));
        Table out =
            Rollup.aggregate(
                t,
                new int[] {0},
                new int[] {0},
                false,
                false,
                new boolean[0],
                new boolean[0],
                GroupByAggregation.sum().onColumn(1))) {
      assertEquals(2, out.getRowCount());
      assertEquals(3, out.getNumberOfColumns());
      try (HostColumnVector keys = out.getColumn(0).copyToHost();
          HostColumnVector gids = out.getColumn(1).copyToHost();
          HostColumnVector sums = out.getColumn(2).copyToHost()) {
        int[] order = sortOrderForOneKeyRollup(keys, gids);
        int d = order[0];
        int g = order[1];
        assertEquals(0L, gids.getLong(d));
        assertFalse(keys.isNull(d));
        assertEquals(7, keys.getInt(d));
        assertEquals(100L, readIntegral(sums, d));
        assertEquals(1L, gids.getLong(g));
        assertTrue(keys.isNull(g));
        assertEquals(100L, readIntegral(sums, g));
      }
    }
  }

  /**
   * Two distinct keys: finest level has two groups; grand total sums both.
   */
  @Test
  void rollupOneIntKey_sum_twoDistinctKeys() {
    try (Table t = new Table(ColumnVector.fromInts(1, 2), ColumnVector.fromInts(10, 20));
        Table out =
            Rollup.aggregate(
                t,
                new int[] {0},
                new int[] {0},
                false,
                false,
                new boolean[0],
                new boolean[0],
                GroupByAggregation.sum().onColumn(1))) {
      assertEquals(3, out.getRowCount());
      try (HostColumnVector keys = out.getColumn(0).copyToHost();
          HostColumnVector gids = out.getColumn(1).copyToHost();
          HostColumnVector sums = out.getColumn(2).copyToHost()) {
        int[] order = sortOrderForOneKeyRollup(keys, gids);
        assertEquals(0L, gids.getLong(order[0]));
        assertEquals(1, keys.getInt(order[0]));
        assertEquals(10L, readIntegral(sums, order[0]));
        assertEquals(0L, gids.getLong(order[1]));
        assertEquals(2, keys.getInt(order[1]));
        assertEquals(20L, readIntegral(sums, order[1]));
        assertEquals(1L, gids.getLong(order[2]));
        assertTrue(keys.isNull(order[2]));
        assertEquals(30L, readIntegral(sums, order[2]));
      }
    }
  }

  /**
   * Same grouping key on two input rows: finest level collapses to one group; grand total matches.
   */
  @Test
  void rollupOneIntKey_sum_duplicateKeyAggregatesWithinFinestLevel() {
    try (Table t = new Table(ColumnVector.fromInts(1, 1), ColumnVector.fromInts(10, 20));
        Table out =
            Rollup.aggregate(
                t,
                new int[] {0},
                new int[] {0},
                false,
                false,
                new boolean[0],
                new boolean[0],
                GroupByAggregation.sum().onColumn(1))) {
      assertEquals(2, out.getRowCount());
      try (HostColumnVector keys = out.getColumn(0).copyToHost();
          HostColumnVector gids = out.getColumn(1).copyToHost();
          HostColumnVector sums = out.getColumn(2).copyToHost()) {
        int[] order = sortOrderForOneKeyRollup(keys, gids);
        int d = order[0];
        int g = order[1];
        assertEquals(0L, gids.getLong(d));
        assertFalse(keys.isNull(d));
        assertEquals(1, keys.getInt(d));
        assertEquals(30L, readIntegral(sums, d));
        assertEquals(1L, gids.getLong(g));
        assertTrue(keys.isNull(g));
        assertEquals(30L, readIntegral(sums, g));
      }
    }
  }

  /**
   * Full {@code ROLLUP(k0, k1)} on two int keys: three grouping levels, {@code SUM} on a third int
   * column.
   */
  @Test
  void rollupTwoIntKeys_sum_sharedFirstKeyRollsUpSecondThenGrand() {
    try (Table t =
            new Table(
                ColumnVector.fromInts(1, 1),
                ColumnVector.fromInts(10, 20),
                ColumnVector.fromInts(100, 200));
        Table out =
            Rollup.aggregate(
                t,
                new int[] {0, 1},
                new int[] {0, 1},
                false,
                false,
                new boolean[0],
                new boolean[0],
                GroupByAggregation.sum().onColumn(2))) {
      assertEquals(4, out.getRowCount());
      assertEquals(4, out.getNumberOfColumns());
      try (HostColumnVector k0 = out.getColumn(0).copyToHost();
          HostColumnVector k1 = out.getColumn(1).copyToHost();
          HostColumnVector gids = out.getColumn(2).copyToHost();
          HostColumnVector sums = out.getColumn(3).copyToHost()) {
        int[] order = sortOrderForTwoKeyRollup(k0, k1, gids);
        // gid=0: (1,10,100), (1,20,200)
        assertRowTwoKey(k0, k1, gids, sums, order[0], 0L, 1, 10, 100L);
        assertRowTwoKey(k0, k1, gids, sums, order[1], 0L, 1, 20, 200L);
        // gid=1: (1,null,300)
        assertRowTwoKey(k0, k1, gids, sums, order[2], 1L, 1, null, 300L);
        // gid=3: (null,null,300) — Spark bitmask for two rolled columns
        assertRowTwoKey(k0, k1, gids, sums, order[3], 3L, null, null, 300L);
      }
    }
  }

  /**
   * ROLLUP on one key: detail rows, grand total, and {@code spark_grouping_id} multiset {@code
   * {0,0,1}} (Spark GpuExpand literals {@code (1L << g) - 1}).
   */
  @Test
  void singleRolledKeyProducesGrandTotalRowAndGroupingId() {
    try (Table t = new Table(ColumnVector.fromInts(1, 2), ColumnVector.fromInts(10, 20));
        Table out =
            Rollup.aggregate(
                t,
                new int[] {0},
                new int[] {0},
                false,
                false,
                new boolean[0],
                new boolean[0],
                GroupByAggregation.sum().onColumn(1))) {
      assertEquals(3, out.getRowCount());
      assertEquals(3, out.getNumberOfColumns());
      assertEquals(1, out.getColumn(0).getNullCount());
      try (HostColumnVector gids = out.getColumn(1).copyToHost()) {
        long[] vals = new long[3];
        for (int i = 0; i < 3; i++) {
          vals[i] = gids.getLong(i);
        }
        Arrays.sort(vals);
        assertEquals(0L, vals[0]);
        assertEquals(0L, vals[1]);
        assertEquals(1L, vals[2]);
      }
      try (HostColumnVector sums = out.getColumn(2).copyToHost()) {
        long sumOfSums = 0;
        for (int i = 0; i < sums.getRowCount(); i++) {
          sumOfSums += readIntegral(sums, i);
        }
        assertEquals(60L, sumOfSums);
      }
    }
  }

  /**
   * One int key ROLLUP with {@code MAX} on the measure: same three-level structure as SUM, values
   * are per-group maxima.
   */
  @Test
  void rollupOneIntKey_max_twoDistinctKeys() {
    try (Table t = new Table(ColumnVector.fromInts(1, 2), ColumnVector.fromInts(10, 20));
        Table out =
            Rollup.aggregate(
                t,
                new int[] {0},
                new int[] {0},
                false,
                false,
                new boolean[0],
                new boolean[0],
                GroupByAggregation.max().onColumn(1))) {
      assertEquals(3, out.getRowCount());
      try (HostColumnVector keys = out.getColumn(0).copyToHost();
          HostColumnVector gids = out.getColumn(1).copyToHost();
          HostColumnVector maxes = out.getColumn(2).copyToHost()) {
        int[] order = sortOrderForOneKeyRollup(keys, gids);
        assertEquals(0L, gids.getLong(order[0]));
        assertEquals(1, keys.getInt(order[0]));
        assertEquals(10L, readIntegral(maxes, order[0]));
        assertEquals(0L, gids.getLong(order[1]));
        assertEquals(2, keys.getInt(order[1]));
        assertEquals(20L, readIntegral(maxes, order[1]));
        assertEquals(1L, gids.getLong(order[2]));
        assertTrue(keys.isNull(order[2]));
        assertEquals(20L, readIntegral(maxes, order[2]));
      }
    }
  }

  /**
   * Two int keys ROLLUP with {@code COUNT} on the measure column (non-null values per Spark GPU
   * count semantics).
   */
  @Test
  void rollupTwoIntKeys_count_sharedFirstKeyRollsUpSecondThenGrand() {
    try (Table t =
            new Table(
                ColumnVector.fromInts(1, 1),
                ColumnVector.fromInts(10, 20),
                ColumnVector.fromInts(100, 200));
        Table out =
            Rollup.aggregate(
                t,
                new int[] {0, 1},
                new int[] {0, 1},
                false,
                false,
                new boolean[0],
                new boolean[0],
                GroupByAggregation.count().onColumn(2))) {
      assertEquals(4, out.getRowCount());
      try (HostColumnVector k0 = out.getColumn(0).copyToHost();
          HostColumnVector k1 = out.getColumn(1).copyToHost();
          HostColumnVector gids = out.getColumn(2).copyToHost();
          HostColumnVector counts = out.getColumn(3).copyToHost()) {
        int[] order = sortOrderForTwoKeyRollup(k0, k1, gids);
        assertRowTwoKey(k0, k1, gids, counts, order[0], 0L, 1, 10, 1L);
        assertRowTwoKey(k0, k1, gids, counts, order[1], 0L, 1, 20, 1L);
        assertRowTwoKey(k0, k1, gids, counts, order[2], 1L, 1, null, 2L);
        assertRowTwoKey(k0, k1, gids, counts, order[3], 3L, null, null, 2L);
      }
    }
  }

  /**
   * Two int keys with both {@code SUM} and {@code COUNT} on the same measure column; output has
   * two trailing aggregate columns.
   */
  @Test
  void rollupTwoIntKeys_sumAndCountOnSameColumn() {
    try (Table t =
            new Table(
                ColumnVector.fromInts(1, 1),
                ColumnVector.fromInts(10, 20),
                ColumnVector.fromInts(100, 200));
        Table out =
            Rollup.aggregate(
                t,
                new int[] {0, 1},
                new int[] {0, 1},
                false,
                false,
                new boolean[0],
                new boolean[0],
                GroupByAggregation.sum().onColumn(2),
                GroupByAggregation.count().onColumn(2))) {
      assertEquals(4, out.getRowCount());
      assertEquals(5, out.getNumberOfColumns());
      try (HostColumnVector k0 = out.getColumn(0).copyToHost();
          HostColumnVector k1 = out.getColumn(1).copyToHost();
          HostColumnVector gids = out.getColumn(2).copyToHost();
          HostColumnVector sums = out.getColumn(3).copyToHost();
          HostColumnVector counts = out.getColumn(4).copyToHost()) {
        int[] order = sortOrderForTwoKeyRollup(k0, k1, gids);
        int r0 = order[0];
        int r1 = order[1];
        int r2 = order[2];
        int r3 = order[3];
        assertRowTwoKey(k0, k1, gids, sums, r0, 0L, 1, 10, 100L);
        assertRowTwoKey(k0, k1, gids, sums, r1, 0L, 1, 20, 200L);
        assertRowTwoKey(k0, k1, gids, sums, r2, 1L, 1, null, 300L);
        assertRowTwoKey(k0, k1, gids, sums, r3, 3L, null, null, 300L);
        assertEquals(1L, readIntegral(counts, r0));
        assertEquals(1L, readIntegral(counts, r1));
        assertEquals(2L, readIntegral(counts, r2));
        assertEquals(2L, readIntegral(counts, r3));
      }
    }
  }

  /**
   * No rolled key positions => one grouping level; {@code spark_grouping_id} is always 0 (still
   * present as a column).
   */
  @Test
  void emptyRolledListIsOrdinaryGroupByWithGroupingIdColumn() {
    try (Table t = new Table(ColumnVector.fromInts(1, 1, 2), ColumnVector.fromInts(10, 20, 30));
        Table out =
            Rollup.aggregate(
                t,
                new int[] {0},
                new int[] {},
                false,
                false,
                new boolean[0],
                new boolean[0],
                GroupByAggregation.count().onColumn(1))) {
      assertEquals(2, out.getRowCount());
      assertEquals(3, out.getNumberOfColumns());
      assertEquals(0, out.getColumn(0).getNullCount());
      try (HostColumnVector gids = out.getColumn(1).copyToHost()) {
        for (int i = 0; i < gids.getRowCount(); i++) {
          assertEquals(0L, gids.getLong(i));
        }
      }
    }
  }

  private static long readIntegral(HostColumnVector col, int row) {
    DType dt = col.getType();
    if (dt.equals(DType.INT64)) {
      return col.getLong(row);
    }
    if (dt.equals(DType.INT32)) {
      return col.getInt(row);
    }
    if (dt.equals(DType.UINT32)) {
      return Integer.toUnsignedLong(col.getInt(row));
    }
    throw new AssertionError("unexpected aggregate output type: " + dt);
  }

  /**
   * Deterministic row order: {@code spark_grouping_id}, then non-null keys before null, then key
   * value (output order from the kernel is not guaranteed).
   */
  private static int[] sortOrderForOneKeyRollup(HostColumnVector keys, HostColumnVector gids) {
    // getRowCount() is long; rollup tests use tiny tables only.
    int n = (int) keys.getRowCount();
    return IntStream.range(0, n)
        .boxed()
        .sorted(
            Comparator.comparingLong((Integer i) -> gids.getLong(i))
                .thenComparing(i -> keys.isNull(i))
                .thenComparingInt(i -> keys.isNull(i) ? 0 : keys.getInt(i)))
        .mapToInt(Integer::intValue)
        .toArray();
  }

  private static int[] sortOrderForTwoKeyRollup(
      HostColumnVector k0, HostColumnVector k1, HostColumnVector gids) {
    int n = (int) k0.getRowCount();
    return IntStream.range(0, n)
        .boxed()
        .sorted(
            Comparator.comparingLong((Integer i) -> gids.getLong(i))
                .thenComparing(i -> k0.isNull(i))
                .thenComparingInt(i -> k0.isNull(i) ? 0 : k0.getInt(i))
                .thenComparing(i -> k1.isNull(i))
                .thenComparingInt(i -> k1.isNull(i) ? 0 : k1.getInt(i)))
        .mapToInt(Integer::intValue)
        .toArray();
  }

  private static void assertRowTwoKey(
      HostColumnVector k0,
      HostColumnVector k1,
      HostColumnVector gids,
      HostColumnVector sums,
      int row,
      long expectedGid,
      Integer expectedK0,
      Integer expectedK1,
      long expectedSum) {
    assertEquals(expectedGid, gids.getLong(row));
    if (expectedK0 == null) {
      assertTrue(k0.isNull(row));
    } else {
      assertFalse(k0.isNull(row));
      assertEquals((int) expectedK0, k0.getInt(row));
    }
    if (expectedK1 == null) {
      assertTrue(k1.isNull(row));
    } else {
      assertFalse(k1.isNull(row));
      assertEquals((int) expectedK1, k1.getInt(row));
    }
    assertEquals(expectedSum, readIntegral(sums, row));
  }
}
