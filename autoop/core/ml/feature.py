class Feature(object):
    """
    Feature class that handles the name and the type of a feature column.
    """

    @property
    def type(self) -> str:
        """
        Getter for the private type attribute

        Returns:
            str: The value of the private attribute type. Returns either
            categorical or numerical depending on which type of feature column
            the Feature class refers to.
        """
        return self._type

    @type.setter
    def type(self, value: str) -> None:
        """
        Setter for the private type attribute. The value that will be assigned
        to the private type attribute can be either numerical or categorical.
        The type attribute refers to the type of feature column the Feature
        class is refering to.

        Args:
            value (str): The value that will be assigned to the private type
            attribute, if the value is valid for the private type value.

        Returns:
            None
        """
        if value not in ["numerical", "categorical"]:
            raise ValueError("Type can only be either numerical or " +
                             "categorical")
        else:
            self._type = value

    def __init__(self, type: str, name: str) -> None:
        """
        Initializer for the Feature class

        Args:
            type (str): The type of the feature column the Feature class is
            refering to. Can be either numerical or categorical. Numerical
            values are continuous values and categorical values can take on a
            limited amount of values.
            name (str): The name of the feature column the Feature class is
            refering to.

        Returns:
            None
        """
        self.type = type
        self.name = name

    def __str__(self) -> str:
        """
        Dunder or magic method that gets called when an instance of the
        Feature class gets formatted in a string. It returns a human readable
        information of the class.

        Returns:
            str: A human readable representation of the class.
        """
        return f"Feature column: ({self.name}, {self.type})"

    def __repr__(self) -> str:
        """
        Dunder or magic method that gets called when repr method is called
        on the Feature object, is called in interactive mode or when items
        in an iterable are displayed. It returns a representation of the class
        in which the class can be reinitialized.

        Returns:
            str: A representation of the Feature class, so it can be
            re-initialized when copied and pasted in a program.
        """
        return f"{self.__class__.__name__}({self.type}, {self.name})"
