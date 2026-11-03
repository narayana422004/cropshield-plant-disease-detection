def format_class_name(class_name):
    """
    Converts PlantVillage class names into
    human-readable text.
    """

    name = class_name.replace("___", " - ")

    name = name.replace("_", " ")

    return name


def format_confidence(confidence):
    """
    Converts a decimal confidence into a percentage.
    """

    return f"{confidence * 100:.2f}%"