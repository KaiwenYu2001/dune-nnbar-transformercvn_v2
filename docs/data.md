# HDF5 input contract

The default dense trainer uses `MinkowskiDataset`, whose name describes the stored
sparse coordinates; it does not require MinkowskiEngine for dense inference.
No raw detector-data converter or sample physics dataset is included.

| Dataset | Expected layout / meaning |
| --- | --- |
| `features` | Float array `[events, max_prongs, feature_dim]` |
| `extra` | Float array `[events, extra_dim]` |
| `prong_mask` | Boolean-compatible array `[events, max_prongs]` |
| `event_target` | Integer class labels `[events]` |
| `prong_target` | Integer labels `[events, max_prongs]`; negative labels are padding |
| `event_compressed_index` | `[events, 2]` lower/upper offsets into event pixel arrays |
| `prong_compressed_index` | `[events, 2]` lower/upper offsets into prong pixel arrays |
| `event_pixels_coordinates` | Integer `[nonzero_pixels, 3]`: local image index, y, x |
| `prong_pixels_coordinates` | Integer `[nonzero_pixels, 3]`: local prong index, y, x |
| `event_pixels_values`, `prong_pixels_values` | Float-compatible `[nonzero_pixels, channels]` |
| `event_pixels_shape`, `prong_pixels_shape` | Stored shape metadata |
| `full_pixels_shape` | `[channels, height, width]` |

Coordinates must be sorted by image index and include each image/prong used by the
mask; the dense conversion infers its batch extent from the final coordinate.
Duplicate coordinates are not supported by the existing indexed assignment.

With `load_full_dataset=false`, pixel arrays must be contiguous, uncompressed HDF5
datasets with valid file offsets, because the reader uses `numpy.memmap`. Use
`load_full_dataset=true` for compressed/chunked input, with enough host RAM.

With `event_current_targets=true`, the reader maps source labels as follows:
labels <= 1 map to class 1; labels > 1 and <= 5 map to class 2; labels > 20 map to
class 3 (NNbar); all remaining labels map to class 0. Verify this mapping against
your source data. Class counts are inferred independently from each split, so
training/validation/test subsets must contain consistent class coverage.

The first prong is always marked valid by the reader. Preprocessing must provide
that prong, and valid prongs must precede padded ones because the trainer truncates
each batch to its maximum count of valid prongs.

Historical split behavior is retained: readers calculate an inclusive maximum
index and then use it as an exclusive slice bound, omitting the final event of
each selected range. Very small ranges can therefore become empty. Correcting
this changes the event membership of existing experiments and is deferred to a
separately validated data-processing change.
