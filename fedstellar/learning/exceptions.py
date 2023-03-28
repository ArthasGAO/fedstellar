# 
# This file is part of the fedstellar framework (see https://github.com/enriquetomasmb/fedstellar).
# Copyright (c) 2022 Enrique Tomás Martínez Beltrán.
# 


class DecodingParamsError(Exception):
    """
    An exception raised when decoding parameters fails.
    """

    pass


class ModelNotMatchingError(Exception):
    """
    An exception raised when parameters do not match with the model.
    """

    pass
