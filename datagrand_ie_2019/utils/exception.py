# -*- coding: UTF-8 -*-


class LengthNotEqualException(Exception):
    """
    corresponded data length is not equal.
    e.g. token count and pos tag count are not equal.
    """

    def __init__(self, message):
        super().__init__(message)


class ModelNotExistedException(Exception):
    """
    model file doesn't exist
    """

    def __init__(self, message):
        super().__init__(message)


class FeatureNotImplemented(Exception):
    """
    Feature is not implemented.
    """

    def __init__(self, message):
        super().__init__(message)


class ParameterError(Exception):
    """
    parameter havs error.
    """

    def __init__(self, message):
        super().__init__(message)


class LabelError(Exception):
    def __init__(self, *args, **kwargs):
        if 'index' in kwargs and 'label' in kwargs and len(kwargs['label']) == 2:
            tmpl = 'position {} and {} are wrong labels {} and {}'
            tmpl_first = 'first label {} is error'
            index = kwargs['index']
            prev_label, label = kwargs['label']
            if not index:
                msg = tmpl_first.format(label)
            else:
                msg = tmpl.format(index - 1, index, prev_label, label)
            super().__init__(msg)
        elif 'name' in kwargs and 'schema' in kwargs:
            tmpl = 'label name {} is not correct in schema {}.'
            super().__init__(tmpl.format(kwargs['name'], kwargs['schema']))
        else:
            super().__init__(*args)


class LabelSchemaError(Exception):
    def __init__(self, *args):
        if len(args) == 1:
            schema = args[0]
            msg = 'label schems {} is not supported.'.format(schema)
            super().__init__(msg)
        else:
            super().__init__(*args)
