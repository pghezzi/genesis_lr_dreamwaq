# Payload specific settings

# Default Container WATER
two_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.0875,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

four_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.0538,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

six_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.03,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

eight_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.01,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

ten_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.015,
    "scale_x":1.6,
    "scale_y":1.6,
    "scale_z":1.6
}

twelve_liters_water_default = {
    "rho": 1000.0,
    "gamma":0.010,
    "mu":0.005,
    "offset":0.0125,
    "scale_x":1.8,
    "scale_y":1.6,
    "scale_z":1.6
}

# Default Container OIL
two_liters_oil_default = {
    "rho": 1000.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.0875,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

four_liters_oil_default = {
    "rho": 1000.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.0538,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

six_liters_oil_default = {
    "rho": 1000.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.03,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

eight_liters_oil_default = {
    "rho": 1000.0,
    "gamma":0.003,
    "mu":0.025,
    "offset":0.01,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

# Default Container GAS
two_liters_gas_default = {
    "rho": 1000.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.0875,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

four_liters_gas_default = {
    "rho": 1000.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.0538,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

six_liters_gas_default = {
    "rho": 1000.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.03,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}

eight_liters_gas_default = {
    "rho": 1000.0,
    "gamma":0.002,
    "mu":0.002,
    "offset":0.01,
    "scale_x":1.6,
    "scale_y":1.4,
    "scale_z":1.4
}


def get_payload_config(payload_type: str, volume: int, container_shape: str = "default"):
    """
    Returns payload configuration dict.

    Args:
        payload_type: {"water", "oil", "gas"}
        volume: payload volume in liters
        container_shape: container geometry identifier (currently unused, stubbed)

    Notes:
        container_shape is reserved for future container-specific configurations.
        Currently, only the "default" shape is supported.
    """

    if container_shape != "default":
        raise NotImplementedError(
            f"container_shape='{container_shape}' is not yet supported"
        )

    if payload_type == "water":
        if volume == 2:
            return two_liters_water_default
        elif volume == 4:
            return four_liters_water_default
        elif volume == 6:
            return six_liters_water_default
        elif volume == 8:
            return eight_liters_water_default
        elif volume == 10:
            return ten_liters_water_default
        elif volume == 12:
            return twelve_liters_water_default

    elif payload_type == "oil":
        if volume == 2:
            return two_liters_oil_default
        elif volume == 4:
            return four_liters_oil_default
        elif volume == 6:
            return six_liters_oil_default
        elif volume == 8:
            return eight_liters_oil_default

    elif payload_type == "gas":
        if volume == 2:
            return two_liters_gas_default
        elif volume == 4:
            return four_liters_gas_default
        elif volume == 6:
            return six_liters_gas_default
        elif volume == 8:
            return eight_liters_gas_default

    else:
        raise ValueError(f"Unsupported payload type: {payload_type}")