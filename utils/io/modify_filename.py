def get_extension_index(filename: str):
    """
    捕获拓展名的分割点位置.
    """

    for index in range(len(filename)):
        rev_ind = len(filename) - index - 1
        if filename[rev_ind] == '.':
            return rev_ind