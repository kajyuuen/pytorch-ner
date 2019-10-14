from torchtext import data, datasets

class Conll2003Dataset(data.Dataset):
    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path=None, fields=None, examples=None, encoding="utf-8", separator="\t", **kwargs):
        if examples is None:
            examples = []
        columns = []

        try:
            with open(path, encoding=encoding) as input_file:
                for line in input_file:
                    line = line.strip()
                    if line.startswith('-DOCSTART-'):  
                        continue
                    if line == "":
                        if columns:
                            examples.append(data.Example.fromlist(columns, fields))
                        columns = []
                    else:
                        for i, column in enumerate(line.split(separator)):
                            if len(columns) < i + 1:
                                columns.append([])
                            columns[i].append(column)
                if columns:
                    examples.append(data.Example.fromlist(columns, fields))
        except TypeError as e:
            pass
        super(Conll2003Dataset, self).__init__(examples, fields, **kwargs)