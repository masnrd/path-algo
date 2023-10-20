import h3
import folium
import os
from PIL import Image


def gradient_color(value: int):
    """
    Return a colour hex value from a value between 0-1
    """
    if value < 0:
        value = 0
    elif value > 1:
        value = 1

    r = int(255 * value)
    b = int(255 * value)
    g = int(255 * value)

    hex_string = ""#{:02x}{:02x}{:02x}"".format(r, b, g)
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
            html=f"<div style='font-size: 8pt'>{i}</div>"
        )
        ).add_to(m)


def visualise_hex_dict_to_map(hexagon_values: dict, m: folium.Map, casualty_locations: dict):
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
        if hexagon_id in casualty_locations:
            # Mark casualty
            folium.Polygon(locations=vertices, color="#FF0000",
                           fill=True).add_to(m)
        else:
            folium.Polygon(locations=vertices, color=color,
                           fill=True, fill_opacity=0.05).add_to(m)


def create_gif(output_filename: str, hexagon_map: dict, hexagon_values: list[dict],
casualty_locations: set, casualty_detected: dict):
    """
    Create a GIF visualization of hexagon values and detected casualties.

    Parameters:
    - output_filename (str): The filename for the resulting GIF.
    - hexagon_map (dict): Mapping of hexagon indices to values.
    - hexagon_values (list): List of hexagon values and indices.
    - casualty_locations (set): Set of locations with casualties.
    - casualty_detected (dict): Mapping of hexagon indices to bool indicating if a casualty was detected.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    import imageio
    from PIL import Image, ImageDraw


    filenames = []

    def create_base_map_image(hex_map_shapes: list[tuple[str, Polygon]],
                              global_xlim: tuple[float, float],
                              global_ylim: tuple[float, float],
                              casualty_locations: set) -> str:
        """
        Create a base map image showing hexagons and casualty locations.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        for _, hex_shape in hex_map_shapes:
            gdf_map = gpd.GeoDataFrame([{'geometry': hex_shape}])
            gdf_map.plot(ax=ax, color="none", edgecolor="k")

        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        ax.axis('off')

        # Plot the entire hexagon map with no fill
        for hex_idx, hex_shape in hex_map_shapes:
            if hex_idx in casualty_locations:
                gdf_map = gpd.GeoDataFrame([{"geometry": hex_shape}])
                gdf_map.plot(ax=ax, color="#FF0000", edgecolor="k")
            else:
                gdf_map = gpd.GeoDataFrame([{"geometry": hex_shape}])
                gdf_map.plot(ax=ax, color="none", edgecolor="k")

        base_filename = os.path.join(os.getcwd(), "base_map.png")
        plt.savefig(base_filename, dpi=300, bbox_inches="tight")
        plt.close()

        return base_filename

    def overlay_hex_on_map(i: int,
                           global_xlim: tuple[float, float],
                           global_ylim: tuple[float, float],
                           previous_filename: str,
                           hexagon_values: list[dict],
                           hex_map_shapes: list[tuple[str, Polygon]],
                           casualty_locations: set,
                           casualty_detected: dict[str, bool]) -> str:
        """
        Overlay a single hexagon onto an existing image.
        """

        # Open the base image (or the previous frame) with PIL
        img = Image.open(previous_filename)
        draw = ImageDraw.Draw(img)

        # Convert the current hexagon to vertices
        hex_idx = hexagon_values[i]["hex_idx"]
        vertices = [(int((pt[1]-global_xlim[0])/(global_xlim[1]-global_xlim[0])*img.width), int((1-(pt[0]-global_ylim[0])/(global_ylim[1]-global_ylim[0]))*img.height)) for pt in h3.h3_to_geo_boundary(hex_idx)]

        if hex_idx in casualty_locations:
            if casualty_detected[hex_idx]:
                color = "green"
            else:
                color = "red"
            draw.polygon(vertices, fill=color, outline="black")
        else:
            draw.polygon(vertices, fill="#D3D3D3", outline="black")

        filename = os.path.join(os.getcwd(), f"frame_{i}.png")
        filenames.append(filename)
        img.save(filename)

        return filename

    hex_map_shapes = [(hex_id, Polygon([(pt[1], pt[0]) for pt in h3.h3_to_geo_boundary(hex_id)])) for hex_id in hexagon_map.keys()]
    all_boundaries = [h3.h3_to_geo_boundary(hv) for hv in hexagon_map.keys()]
    all_lons = [pt[1] for boundary in all_boundaries for pt in boundary]
    all_lats = [pt[0] for boundary in all_boundaries for pt in boundary]
    global_xlim = (min(all_lons), max(all_lons))
    global_ylim = (min(all_lats), max(all_lats))

    # Step 1: Create the base map image
    base_filename = create_base_map_image(hex_map_shapes, global_xlim, global_ylim, casualty_locations)

    # Step 2: Overlay hexagons on the base map
    previous_filename = base_filename
    for i, hexagon in enumerate(hexagon_values):
        previous_filename = overlay_hex_on_map(i, global_xlim, global_ylim, previous_filename, hexagon_values, hex_map_shapes, casualty_locations, casualty_detected)

    with imageio.get_writer(output_filename, mode="I", duration=1) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)


