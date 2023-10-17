import h3
import folium

# Utils for visualisation

# Return a colour hex value from a value between 0-1
def gradient_color(value):
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
def add_hex_to_map(hexagon_values, m):
    keys = list(hexagon_values.keys())
    num_keys = len(keys)

    for i, hexagon_id in enumerate(hexagon_values):
        vertices = h3.h3_to_geo_boundary(hexagon_id)
        color = color = gradient_color(hexagon_values[hexagon_id])
        folium.Polygon(locations=vertices, color=color, fill=True, fill_opacity=0.6).add_to(m)
        folium.map.Marker(h3.h3_to_geo(hexagon_id),
                icon=folium.DivIcon(
                    icon_size=(10,10),
                    icon_anchor=(5,14),
                    html=f'<div style="font-size: 8pt">{i}</div>'
                )
                ).add_to(m)