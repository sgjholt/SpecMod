def gps2dist_azimuth(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    a: float = ...,
    f: float = ...,
) -> tuple[float, float, float]:
    """Distance in **metres**, azimuth, back-azimuth — in that order.

    Not symmetric in floating point: swapping the two points changes the last
    bit of the distance. `preprocess.set_stream_distance` depends on that and
    says so.
    """
