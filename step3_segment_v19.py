import rasterio
import numpy as np
import cv2
import math
import os
import pickle
import gc
import shapefile
from shapely.geometry import box, LineString, Point
from shapely.ops import unary_union
from scipy.spatial import distance
from ultralytics import YOLO
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

output_path = 'detections'

orthos = [
    #'ortho1.tif',
    #'ortho2.tif',
    'VDN2024_clipped_3.tif',
]

thalweg_shapefile_path = 'shapefiles/VDNthalwegLV95.shp'

model = YOLO('yolov8_drywood_segmentation_320p_extralarge_200epochs_plus_large_samples_7aug/weights/best.pt')  # load a custom model

pixel_adjustment = 0.6226
pixel_adjustment = 1

confidence_segmentation = 0.5

chosen_percentage = 30

def crop_bounding_box(orthomosaic_path, x_min_real, y_min_real, x_max_real, y_max_real, output_crop_path):
    with rasterio.open(orthomosaic_path) as src:
        transform = src.transform

        x_min, y_min = ~transform * (x_min_real, y_min_real)
        x_max, y_max = ~transform * (x_max_real, y_max_real)

        x_min = int(np.clip(x_min, 0, src.width - 1))
        y_min = int(np.clip(y_min, 0, src.height - 1))
        x_max = int(np.clip(x_max, 0, src.width - 1))
        y_max = int(np.clip(y_max, 0, src.height - 1))

        if y_min > y_max:
            y_min, y_max = y_max, y_min

        if x_min > x_max:
            x_min, x_max = x_max, x_min

        ortho_crop = src.read(window=((y_min, y_max + 1), (x_min, x_max + 1)))
        crop_transform = rasterio.windows.transform(((y_min, y_max + 1), (x_min, x_max + 1)), src.transform)

        if ortho_crop.shape[1] == 0 or ortho_crop.shape[2] == 0:
            raise ValueError(f"Cropped area has zero size: shape={ortho_crop.shape}")

        meta = src.meta
        meta.update({"height": ortho_crop.shape[1], "width": ortho_crop.shape[2], "transform": crop_transform})
        with rasterio.open(output_crop_path, 'w', **meta) as dst:
            dst.write(ortho_crop)

        return ortho_crop, crop_transform

def calculate_polygon_area(coordinates):
    n = len(coordinates)
    if n < 3:
        raise ValueError("A polygon must have at least 3 vertices.")

    if isinstance(coordinates, np.ndarray):
        coordinates = coordinates.tolist()
    
    coordinates.append(coordinates[0])

    area = 0
    for i in range(n):
        x_i, y_i = coordinates[i]
        x_next, y_next = coordinates[i + 1]
        area += (x_i * y_next - y_i * x_next)
    
    area = abs(area) / 2
    return area

def segment_wood_pixels(ortho_crop, detection_id, cropped_folder, orientation):
    ortho_crop_rgb = np.moveaxis(ortho_crop, 0, -1)

    if ortho_crop_rgb.shape[2] == 4:
        ortho_crop_rgb = cv2.cvtColor(ortho_crop_rgb, cv2.COLOR_RGBA2RGB)

    ortho_crop_rgb = np.ascontiguousarray(ortho_crop_rgb, dtype=np.uint8)

    if ortho_crop_rgb.shape[0] == 0 or ortho_crop_rgb.shape[1] == 0:
        raise ValueError(f"Cropped image for detection_id {detection_id} has invalid dimensions: {ortho_crop_rgb.shape}")

    results = model.predict(ortho_crop_rgb, verbose=False, conf=confidence_segmentation)

    wood_pixels_count = 0
    PIXpolygons = []
    cumulative_mask = np.zeros(ortho_crop_rgb.shape[:2], dtype=np.uint8)  # To track all counted pixels

    for result in results:
        masks = result.masks

        if masks is None:
            print(f"No masks detected for detection_id {detection_id}.")
            continue

        if len(masks.xy) > 1:
            try:
                # Assuming masks.xy is a list of numpy arrays representing the vertices of each polygon mask
                masks_xy = masks.xy

                # Convert each mask to a shapely Polygon object
                polygons = [Polygon(mask) for mask in masks_xy]

                # Perform the union operation to merge all polygons
                merged_polygon = unary_union(polygons)

                # If you want to get the merged polygon's points (x, y coordinates)
                masks_xy = [np.array(merged_polygon.exterior.coords)]
            except:
                continue

        else:
            #print(masks.xy)
            masks_xy = masks.xy

        for mask in masks_xy:
            mask_image = np.zeros(ortho_crop_rgb.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_image, [np.array(mask, dtype=np.int32)], 1)
            
            # Count new pixels by checking where cumulative_mask is still 0
            new_pixels = np.sum((mask_image == 1) & (cumulative_mask == 0))
            wood_pixels_count += new_pixels
            
            # Update the cumulative mask
            cumulative_mask = np.maximum(cumulative_mask, mask_image)
            
            PIXpolygons.append(mask)

        image_with = cv2.polylines(ortho_crop_rgb, [np.int32(mask) for mask in masks.xy], isClosed=True, color=(255, 0, 0), thickness=1)
        cv2.imwrite(os.path.join(cropped_folder, f'detection_{detection_id}_crop_segmented_'+str(orientation)+'.jpg'), image_with)

    return wood_pixels_count, PIXpolygons


def calculate_volume(length, diameter):
    radius = diameter / 2
    volume = length * math.pi * (radius ** 2)
    return volume

def find_groups(detections_info):
    boxes = [box(d[4], d[5], d[6], d[7]) for d in detections_info]
    groups = []
    group_indices = []

    for i, b in enumerate(boxes):
        found_group = False
        for j, group in enumerate(groups):
            if any(b.intersects(other) for other in group):
                groups[j].append(b)
                group_indices[j].append(i)
                found_group = True
                break
        if not found_group:
            groups.append([b])
            group_indices.append([i])

    merged_groups = [unary_union(group) for group in groups]
    return merged_groups, group_indices

# Function to determine orientation based on edge densities
def determine_orientation(gray):
    # Define Sobel kernels for diagonal edge detection
    sobel_diag1 = np.array([[ 0,  1, 2],
                            [-1,  0, 1],
                            [-2, -1, 0]])
    sobel_diag12 = np.array([[ 0, -1, -2],
                             [ 1,  0, -1],
                             [ 2,  1,  0]])

    sobel_diag2 = np.array([[2,  1,  0],
                            [1,  0, -1],
                            [0, -1, -2]])
    sobel_diag22 = np.array([[-2, -1,  0],
                             [-1,  0,  1],
                             [ 0,  1,  2]])

    # Apply Sobel edge detection in diagonal directions
    edge_diag1 = cv2.filter2D(gray, -1, sobel_diag1)
    edge_diag12 = cv2.filter2D(gray, -1, sobel_diag12)

    edge_diag2 = cv2.filter2D(gray, -1, sobel_diag2)
    edge_diag22 = cv2.filter2D(gray, -1, sobel_diag22)

    n_white_pix_1 = np.sum(edge_diag1 > 100) + np.sum(edge_diag12 > 100)
    n_white_pix_2 = np.sum(edge_diag2 > 100) + np.sum(edge_diag22 > 100)

    if n_white_pix_1 > n_white_pix_2:
        return "nwse"  # Top left to bottom right
    else:
        return "nesw"  # Top right to bottom left

def load_shapefile(shapefile_path):
    sf = shapefile.Reader(shapefile_path)
    lines = []
    for shape in sf.shapes():
        lines.append(LineString(shape.points))
    return lines

def calculate_angle_between_lines(line1, line2):
    # Calculate direction vectors of the lines
    v1 = np.array([line1.coords[1][0] - line1.coords[0][0], line1.coords[1][1] - line1.coords[0][1]])
    v2 = np.array([line2.coords[1][0] - line2.coords[0][0], line2.coords[1][1] - line2.coords[0][1]])

    # Calculate the angle between the direction vectors
    angle = np.arccos(np.clip(np.abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    return np.degrees(angle)

def find_closest_line(point, lines):
    closest_line = None
    min_distance = float('inf')
    for line in lines:
        dist = line.distance(point)
        if dist < min_distance:
            min_distance = dist
            closest_line = line
    return closest_line

def project_point_on_line(point, line):
    return line.interpolate(line.project(point))

def calculate_distance_along_line(main_line, point):
    # Project the point onto the main line
    projected_point = project_point_on_line(point, main_line)
    
    # Calculate the cumulative distance along the main line to the projected point
    distance_along_line = main_line.project(projected_point)
    
    return distance_along_line

def process_detection(detection_info, orthomosaic_path, cropped_folder, output_info_txt_path, group_id, group_total_pixels, group_unique_pixels, thalweg_lines, main_line):
    detection_id, score, x_center, y_center, x_min_real, y_min_real, x_max_real, y_max_real, length = detection_info

    round_score = int(round(score * 100, 0))

    output_crop_path = os.path.join(cropped_folder, f"detection_{detection_id}_crop.tif")
    ortho_crop, crop_transform = crop_bounding_box(orthomosaic_path, x_min_real, y_min_real, x_max_real, y_max_real, output_crop_path)

    # Resize the bounding box to a square
    ortho_crop_resized = cv2.resize(np.moveaxis(ortho_crop, 0, -1), (256, 256), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray_crop = cv2.cvtColor(ortho_crop_resized, cv2.COLOR_RGB2GRAY)

    # Determine orientation
    orientation = determine_orientation(gray_crop)

    crop_pickle_path = os.path.join(cropped_folder, 'detection_' + str(detection_id) + '_' + str(round_score) + 'perc_crop_info.pkl')
    crop_info = (orthomosaic_path, x_min_real, y_min_real, x_max_real, y_max_real)

    with open(crop_pickle_path, 'wb') as file:
        pickle.dump(crop_info, file)

    resolution_x = crop_transform[0]
    resolution_y = abs(crop_transform[4])
    width_meters = resolution_x * (ortho_crop.shape[2] - 1)
    height_meters = resolution_y * (ortho_crop.shape[1] - 1)
    bbox_surface_area = width_meters * height_meters

    total_pixels = (ortho_crop.shape[2]) * (ortho_crop.shape[1])
    pixel_size_square_meters = bbox_surface_area / total_pixels

    wood_pixels_count, PIXpolygons = segment_wood_pixels(ortho_crop, detection_id, cropped_folder, orientation)

    polygons_pickle_path = os.path.join(cropped_folder, 'detection_' + str(detection_id) + '_' + str(round_score) + 'perc_polygons_'+str(orientation)+'.pkl')

    with open(polygons_pickle_path, 'wb') as file:
        pickle.dump(PIXpolygons, file)

    wood_surface_area = wood_pixels_count * pixel_size_square_meters
    diag_length_real = math.sqrt((x_max_real - x_min_real) ** 2 + (y_max_real - y_min_real) ** 2)
    if (length - diag_length_real) / length > 0.05:
        print('The 2 calculated lengths do not seem to match!!!!!!!!!!!!!!!')

    diameter = wood_surface_area / diag_length_real
    volume = calculate_volume(length, diameter)

    # Handle the zero division case
    if group_total_pixels == 0:
        adjusted_volume = 0
    else:
        adjusted_volume = volume * (group_unique_pixels / group_total_pixels)

    # Find the closest line segment and calculate the angle if thalweg_lines is not None
    if thalweg_lines is not None:
        detection_center = Point(x_center, y_center)
        closest_line = find_closest_line(detection_center, thalweg_lines)
        if orientation == "nwse":
            piece_line = LineString([(x_min_real, y_min_real), (x_max_real, y_max_real)])
        else:
            piece_line = LineString([(x_max_real, y_min_real), (x_min_real, y_max_real)])
        angle = calculate_angle_between_lines(piece_line, closest_line)

        # Calculate the distance along the main line
        distance_on_main_line = calculate_distance_along_line(main_line, detection_center)
    else:
        angle = '-'
        distance_on_main_line = '-'

    return detection_id, score, x_center, y_center, x_min_real, y_min_real, x_max_real, y_max_real, length, int(wood_pixels_count), diameter*pixel_adjustment, volume*pixel_adjustment, group_id, group_total_pixels, group_unique_pixels, adjusted_volume*pixel_adjustment, orientation, angle, distance_on_main_line

def process_detections_from_txt(input_txt_path, orthomosaic_path, output_info_txt_path, cropped_folder, thalweg_lines, main_line):
    if not os.path.exists(cropped_folder):
        os.makedirs(cropped_folder)

    detections_info = []
    with open(input_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 9:
                print(f"Skipping malformed line: {line}")
                continue
            try:
                detection_info = tuple(map(float, parts))
                detections_info.append(detection_info)
            except ValueError as ve:
                print(f"Error parsing line: {line}. Error: {ve}")
                continue

    merged_groups, group_indices = find_groups(detections_info)

    results = []
    for group_id, (group, indices) in enumerate(zip(merged_groups, group_indices)):
        print('')
        print('JAM ID')
        print(group_id)
        x_min, y_min, x_max, y_max = group.bounds
        output_crop_path = os.path.join(cropped_folder, f"group_{group_id}_crop.tif")
        ortho_crop, crop_transform = crop_bounding_box(orthomosaic_path, x_min, y_min, x_max, y_max, output_crop_path)

        ortho_crop_rgb = np.moveaxis(ortho_crop, 0, -1)
        if ortho_crop_rgb.shape[2] == 4:
            ortho_crop_rgb = cv2.cvtColor(ortho_crop_rgb, cv2.COLOR_RGBA2RGB)
        ortho_crop_rgb = np.ascontiguousarray(ortho_crop_rgb, dtype=np.uint8)

        group_total_pixels = 0
        unique_pixel_set = set()

        for idx in indices:
            detection_info = detections_info[idx]
            detection_id = int(detection_info[0])

            round_score = int(round(detection_info[1] * 100, 0))
            ortho_crop_segment, crop_transform = crop_bounding_box(orthomosaic_path, detection_info[4], detection_info[5], detection_info[6], detection_info[7], output_crop_path)
            ortho_crop_rgb_segment = np.moveaxis(ortho_crop_segment, 0, -1)
            if ortho_crop_rgb_segment.shape[2] == 4:
                ortho_crop_rgb_segment = cv2.cvtColor(ortho_crop_rgb_segment, cv2.COLOR_RGBA2RGB)
            ortho_crop_rgb_segment = np.ascontiguousarray(ortho_crop_rgb_segment, dtype=np.uint8)

            # Ensure the image has valid dimensions
            if ortho_crop_rgb_segment.shape[0] == 0 or ortho_crop_rgb_segment.shape[1] == 0:
                print(f"Skipping detection_id {detection_id} due to invalid dimensions: {ortho_crop_rgb_segment.shape}")
                continue

            results_segment = model.predict(ortho_crop_rgb_segment, verbose=False, conf=confidence_segmentation)
            wood_pixels_count_segment = 0
            for result in results_segment:
                masks = result.masks
                #print(masks.xy)
                if masks is None:
                    continue

                if len(masks.xy) > 1:
                    try:

                        # Assuming masks.xy is a list of numpy arrays representing the vertices of each polygon mask
                        masks_xy = masks.xy

                        # Convert each mask to a shapely Polygon object
                        polygons = [Polygon(mask) for mask in masks_xy]

                        # Perform the union operation to merge all polygons
                        merged_polygon = unary_union(polygons)

                        # If you want to get the merged polygon's points (x, y coordinates)
                        masks_xy = [np.array(merged_polygon.exterior.coords)]

                        # Merge all arrays in the list into one array
                        #masks_xy = masks.xy
                        # Merge the two masks into one array
                        #merged_points = np.vstack(masks_xy)

                        # Compute the convex hull
                        #hull = ConvexHull(merged_points)

                        # Get the points of the convex hull (the envelope)
                        #envelope_points = merged_points[hull.vertices]

                        # Assign the envelope points back to masks.xy
                        #masks_xy = [envelope_points]
                    except:
                        continue

                else:
                    #print(masks.xy)
                    masks_xy = masks.xy

                for mask in masks_xy:
                    mask_image = np.zeros(ortho_crop_rgb.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_image, [np.array(mask, dtype=np.int32)], 1)
                    wood_pixels_count_segment += np.sum(mask_image)
                    #PIXpolygons_segment = mask

                    for y in range(mask_image.shape[0]):
                        for x in range(mask_image.shape[1]):
                            if mask_image[y, x] > 0:
                                unique_pixel_set.add((x, y))

            group_total_pixels += wood_pixels_count_segment

        group_unique_pixels = len(unique_pixel_set)

        for idx in indices:
            detection_info = detections_info[idx]
            result = process_detection(detection_info, orthomosaic_path, cropped_folder, output_info_txt_path, group_id, group_total_pixels, group_unique_pixels, thalweg_lines, main_line)
            results.append(result)
            gc.collect()

    with open(output_info_txt_path, 'w') as f:
        for result in results:
            f.write(','.join(map(str, result)) + '\n')

    with open(output_info_txt_path.replace('.txt','.csv'), 'w') as f:
        f.write('detection_id,confidence,Xcenter,Ycenter,Xmin,Ymax,Xmax,Ymin,length,wood_pixels_count,diameter,volume,group_id,group_total_pixels,group_unique_pixels,adjusted_volume,orientation,angle,distance_on_main_line\n')
        for result in results:
            f.write(','.join(map(str, result)) + '\n')

if os.path.exists(thalweg_shapefile_path):
    thalweg_lines = load_shapefile(thalweg_shapefile_path)
    print(thalweg_lines)
    main_line = thalweg_lines[0]  # Assuming the first line is the main line
    # Merge all segments into one main line if there are multiple segments
    if len(thalweg_lines) > 1:
        main_line = LineString([point for line in thalweg_lines for point in line.coords])
else:
    thalweg_lines = None
    main_line = None
    print("Shapefile not found. Angles and distances will be marked as '-' in the output.")

def process(ortho):
    input_txt_path = os.path.join(output_path,ortho.replace('.tif','_detections_conf_'+str(chosen_percentage)+'perc.txt'))
    cropped_folder = input_txt_path.replace('.txt','_cropped')
    output_info_txt_path = os.path.join(output_path,ortho.replace('.tif','_detections_conf_'+str(chosen_percentage)+'perc_info.txt'))
    process_detections_from_txt(input_txt_path, ortho, output_info_txt_path, cropped_folder, thalweg_lines, main_line)

for ortho in orthos:
    process(ortho)
