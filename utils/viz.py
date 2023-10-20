import h3
import folium

# Utils for visualisation

# Return a colour hex value from a value between 0-1


def gradient_color(value: int):
    if value < 0:
        value = 0
    elif value > 1:
        value = 1

    r = int(255 * value)
    b = int(255 * value)
    g = int(255 * value)

    hex_string = '#{:02x}{:02x}{:02x}'.format(r, b, g)
    return hex_string

# Add H3 hexagons as polygons to the map, m


def add_hex_to_map(hexagon_values: list, m: folium.Map):
    """
    Visualise the path based on index

    Parameters:
    - hexagon_values: list of dictionaries with hex_idx and logical clock of when it is traversed
    - m: folium map

    Returns:
    """
    num_keys = len(hexagon_values)

    for i in range(len(hexagon_values)):
        vertices = h3.h3_to_geo_boundary(hexagon_values[i]["hex_idx"])
        color = color = gradient_color(
            hexagon_values[i]["step_count"]/num_keys)
        folium.Polygon(locations=vertices, color=color,
                       fill=True, fill_opacity=0.6).add_to(m)
        folium.map.Marker(h3.h3_to_geo(hexagon_values[i]["hex_idx"]),
                          icon=folium.DivIcon(
            icon_size=(10, 10),
            icon_anchor=(5, 14),
            html=f'<div style="font-size: 8pt">{i}</div>'
        )
        ).add_to(m)


def visualise_hex_dict_to_map(hexagon_values: dict, m: folium.Map):
    """
    Visualise the probability map.

    Parameters:
    - hexagon_values: dictionary of hex_idx and probability values
    - m: folium map

    Returns:
    """
    keys = list(hexagon_values.keys())
    num_keys = len(keys)

    for i, hexagon_id in enumerate(hexagon_values):
        vertices = h3.h3_to_geo_boundary(hexagon_id)
        color = color = gradient_color(hexagon_values[hexagon_id])
        folium.Polygon(locations=vertices, color=color,
                       fill=True, fill_opacity=0.05).add_to(m)
