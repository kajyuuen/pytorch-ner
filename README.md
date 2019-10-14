# Input data format

```
token label
token label
token label
```

Unlabeled token's tag is `NOANNOTATION`.

# Train

## BiLSTM-CRF

```
$ python train.py config/lample.json
```

## BiLSTM-Fuzzy-CRF


```
$ python train.py config/partial.json
```

## Simple Approach

This approach convert `NOANNOTATION` tag to `O` tag.

```
$ python train.py config/simple.json
```

# Labeling tools

## Delete label

```
$ python remove_annotation.py <Input file path> <Output file path> --entity_keep_ratio 0.4
```

## Create Dictionary

```
$ python dictionary_create.py <Input file path> <Output file path>
```

## Annotation using Dictionary

```
$ python dictionary_labeling.py <Input file path> <Dictionary file path> <Output file path>
```

## References

- https://github.com/kmkurn/pytorch-crf
- https://github.com/threelittlemonkeys/lstm-crf-pytorch
