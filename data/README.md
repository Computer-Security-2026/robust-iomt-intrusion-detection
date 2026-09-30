# Dataset

The planned experiments use the CICIoMT2024 dataset referenced by the original paper and implementation.

Do not commit the full dataset to this repository. Document the following before experiments begin:

- official download location and access date;
- exact files used;
- file checksums;
- selected features and target labels;
- cleaning and preprocessing steps;
- training, validation, and test split strategy; and
- any sampling used to make local experiments manageable.

The split must prevent the same or near-duplicate records from leaking across training and test data. The team should record whether splitting is performed by row, source file, device, attack session, or time period.
