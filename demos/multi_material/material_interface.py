import firedrake as fd


def sharp_interface(level_set, material_value, method):
    ls = level_set.pop()

    if level_set:  # Directly specify material value on only one side of the interface
        return fd.conditional(
            ls > 0.5,
            material_value.pop(),
            sharp_interface(level_set, material_value, method),
        )
    else:  # Final level set; specify values for last two materials
        return fd.conditional(ls < 0.5, *material_value)


def diffuse_interface(level_set, material_value, method):
    ls = fd.max_value(fd.min_value(level_set.pop(), 1), 0)

    if level_set:
        match method:
            case "arithmetic":
                return material_value.pop() * ls + diffuse_interface(
                    level_set, material_value, method
                ) * (1 - ls)
            case "geometric":
                return material_value.pop() ** ls * diffuse_interface(
                    level_set, material_value, method
                ) ** (1 - ls)
            case "harmonic":
                return 1 / (
                    ls / material_value.pop()
                    + (1 - ls) / diffuse_interface(level_set, material_value, method)
                )
    else:
        match method:
            case "arithmetic":
                return material_value[0] * (1 - ls) + material_value[1] * ls
            case "geometric":
                return material_value[0] ** (1 - ls) * material_value[1] ** ls
            case "harmonic":
                return 1 / ((1 - ls) / material_value[0] + ls / material_value[1])
