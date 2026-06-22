class Args():
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __str__(self):
        return str(self.__dict__)