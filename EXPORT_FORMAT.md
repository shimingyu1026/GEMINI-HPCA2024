# Export Format

Gemini now exports two extra artifacts after `DP_search()` completes:

- `chiplet_events.json`
- `scalesim_topology.csv`

By default they are written to:

- `output/exports/`

If `stschedule` is launched with an output log path such as:

```bash
./build/stschedule tmp/run/result.log
```

the export directory becomes:

- `tmp/run/exports/`

## Export Scope

The exporter keeps the original Gemini mapping result, then aggregates it to the big-chiplet level.

- Intra-chiplet transfers are dropped.
- Only `DRAM <-> chiplet` and `chiplet <-> chiplet` transfers are kept.
- Compute events are aggregated per chiplet.
- `ConvLayer`, `GroupConvLayer`, and `FCLayer` are written to `scalesim_topology.csv`.
- Other layer types are ignored in the CSV export.

Time order is reconstructed from each core's workload list and then merged into a chiplet-level timeline. This is a chiplet-level execution order, not a cycle-accurate replay of intra-chiplet details.

## `chiplet_events.json`

Top-level structure:

```json
{
  "metadata": {
    "network_name": "darknet19",
    "xlen": 4,
    "ylen": 4,
    "x_cut": 2,
    "y_cut": 2,
    "x_step": 2,
    "y_step": 2,
    "chiplet_count": 4,
    "source": "Gemini per-core IR"
  },
  "chiplets": [
    {
      "chiplet_id": 0,
      "chiplet_x": 0,
      "chiplet_y": 0,
      "core_ids": [1, 2, 7, 8],
      "events": []
    }
  ]
}
```

Each chiplet event list is sorted by:

- event time
- event type rank: `read`, `compute`, `write`

Each event also gets a local `seq` index inside that chiplet timeline.

### `read` / `write` events

Fields:

- `type`: `read` or `write`
- `chiplet_id`: local chiplet id
- `time`: chiplet event timestamp
- `layer_name`: layer name in Gemini
- `tensor_type`
  - `weight`
  - `activation`
  - `ofmap`
- `bytes`: transfer size in bytes
- `peer.kind`
  - `dram`
  - `chiplet`
- `peer.chiplet_id`: present only when `peer.kind == "chiplet"`
- `transfer_ids`: merged Gemini transfer ids
- `workload_ids`: related workload ids
- `seq`: event order inside the chiplet

Rules:

- Same-chiplet core-to-core transfers are not exported.
- `read` events come from:
  - `DRAM -> chiplet`
  - `chiplet A -> chiplet B`
- `write` events come from:
  - `chiplet -> DRAM`
  - `chiplet A -> chiplet B`

### `compute` events

Fields:

- `type`: `compute`
- `chiplet_id`
- `time_start`
- `time_end`
- `duration`
- `layer_name`
- `layer_type`
- `partition_count`: number of merged core workloads
- `fragmented`: whether the merged region is not a dense bounding box
- `ofmap_elements`: exact merged output elements
- `ofmap_bbox_elements`: size of the merged output bounding box
- `ofmap.lower` / `ofmap.upper`
- `ifmap.lower` / `ifmap.upper`
- `workload_ids`
- `core_ids`
- `layer_shape`
- `seq`

`layer_shape.kind` is one of:

- `conv2d`
- `group_conv2d`
- `fc`
- `other`

For grouped convolution, `layer_shape` also includes `groups`.

## `scalesim_topology.csv`

Header:

```csv
Layer name,IFMAP Height,IFMAP Width,Filter Height,Filter Width,Channels,Num Filter,Stride Height,Stride Width,Batch Size,
```

This file is directly consumable by SCALE-Sim v3 as a conv-topology CSV.

### Row naming

Base pattern:

```text
chiplet<chiplet_id>_<layer_name>_t<start_time>
```

Grouped convolution rows append a group suffix:

```text
chiplet0_encoder_QK_t19346_g2
```

There is intentionally no per-workload suffix such as `_w17`. The CSV is chiplet-aggregated.

### Layer mapping rules

- `ConvLayer`
  - one aggregated conv row per chiplet compute event
- `FCLayer`
  - exported as conv-style rows, matching Gemini's internal FC-as-conv representation
- `GroupConvLayer`
  - split into multiple conv rows, one row per touched group

Ignored in CSV:

- `PoolingLayer`
- `EltwiseLayer`
- `PTPLayer`
- `TransposeLayer`
- any other non-conv-like layer

## Running with SCALE-Sim v3

Example:

```bash
PYTHONPATH=/Users/smy/Proj/scale-sim-v3 \
python3 /Users/smy/Proj/scale-sim-v3/scale.py \
  -c /Users/smy/Proj/scale-sim-v3/configs/scale.cfg \
  -l /Users/smy/Proj/scale-sim-v3/layouts/conv_nets/test.csv \
  -t ./output/exports/scalesim_topology.csv \
  -p ./tmp/scalesim_check
```

Note:

- `scale.py` requires `-l` to point to an existing layout file even when custom layout is disabled.
