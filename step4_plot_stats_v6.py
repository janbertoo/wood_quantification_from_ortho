import pandas as pd
import matplotlib.pyplot as plt
import os

ortho = 'VDN2024_clipped_3.tif'

chosen_conf_percentage = 30

# Read the data from the text file
file_path = os.path.join('detections',ortho.split('.tif')[0]+'_detections_conf_'+str(chosen_conf_percentage)+'perc_info.txt')  # Replace with your actual file path

output_path = 'images'

if not os.path.exists(output_path):
    os.makedirs(output_path) 

column_names = [
    'detection_id', 'confidence', 'x_center', 'y_center', 'x_min', 'y_max',
    'x_max', 'y_min', 'length', 'wood_pixel_count', 'diameter', 'volume', 'jam_number',
    'total_pixels', 'unique_pixels', 'adjusted_volume', 'orientation',
    'orientation_angle', 'distance_from_origin'
]

data = pd.read_csv(file_path, header=None, names=column_names)

# Calculate the total adjusted volume
total_adjusted_volume = data['adjusted_volume'].sum()
print(f'Total Adjusted Volume: {total_adjusted_volume} m^3')

# Bin the distance from origin in brackets of 50 meters
data['distance_bin'] = (data['distance_from_origin'] // 50) * 50

# Group by the distance bin and sum the adjusted volumes
adjusted_volume_by_distance = data.groupby('distance_bin')['adjusted_volume'].sum().reset_index()

# Plotting Volume vs Distance
plt.figure(figsize=(12, 8))
plt.bar(adjusted_volume_by_distance['distance_bin'], adjusted_volume_by_distance['adjusted_volume'], width=40, align='edge', edgecolor='black')
plt.xlabel('Distance from Origin (m)')
plt.ylabel('Volume (m^3)')
plt.title(f'Volume vs. Distance from Origin (Total Volume: {total_adjusted_volume:.2f} m^3)')
plt.xticks(adjusted_volume_by_distance['distance_bin'])
plt.grid(True)
plt.savefig(os.path.join(output_path,'volume_vs_distance.png'))
plt.show()

# Define orientation angle bins
bins = [0, 22.5, 67.5, 90]
labels = ['0-22.5', '22.5-67.5', '67.5-90']
data['orientation_bin'] = pd.cut(data['orientation_angle'], bins=bins, labels=labels, include_lowest=True)

# Group by orientation bins and count occurrences
orientation_counts = data['orientation_bin'].value_counts().sort_index()

# Plotting Orientation Angles
plt.figure(figsize=(10, 6))
plt.bar(orientation_counts.index, orientation_counts.values, color='skyblue', edgecolor='black')
plt.xlabel('Orientation Angle Bins (degrees)')
plt.ylabel('Count')
plt.title('Orientation Angles Distribution')
plt.grid(True)
plt.savefig(os.path.join(output_path,'orientation_angles_distribution.png'))
plt.show()

# Plotting Volume Distribution
plt.figure(figsize=(10, 6))
plt.hist(data['adjusted_volume'], bins=30, color='green', edgecolor='black')
plt.xlabel('Volume (m^3)')
plt.ylabel('Frequency')
plt.title('Volume Distribution')
plt.grid(True)
plt.savefig(os.path.join(output_path,'volume_distribution.png'))
plt.show()

# Check and adjust length values if necessary (assuming length should be in meters)
if data['length'].max() > 100:  # Assuming no piece is longer than 100 meters
    data['length'] = data['length'] / 1000

print(data['length'])

# Plotting Length Distribution
plt.figure(figsize=(10, 6))
plt.hist(data['length'], bins=30, color='orange', edgecolor='black')
plt.xlabel('Length (m)')
plt.ylabel('Frequency')
plt.title('Length Distribution')
plt.grid(True)
plt.savefig(os.path.join(output_path,'length_distribution.png'))
plt.show()

# Count the number of detections per jam
detection_counts = data['jam_number'].value_counts().reset_index()
detection_counts.columns = ['jam_number', 'count']

# Filter out jams with fewer than 2 detections
valid_jams = detection_counts[detection_counts['count'] >= 2]['jam_number']

# Filter the original data to include only valid jams
filtered_data = data[data['jam_number'].isin(valid_jams)]

# Group by jam number and sum the adjusted volumes for valid jams
volume_by_jam = filtered_data.groupby('jam_number')['adjusted_volume'].sum().reset_index()

# Sort by jam number
volume_by_jam = volume_by_jam.sort_values('jam_number')

# Plotting Total Volume per Jam with at least 2 detections
plt.figure(figsize=(12, 8))
plt.bar(volume_by_jam['jam_number'].astype(str), volume_by_jam['adjusted_volume'], color='purple', edgecolor='black')
plt.xlabel('Jam Number')
plt.ylabel('Total Volume (m^3)')
plt.title('Total Volume per Jam')
#plt.xticks(rotation=90)
plt.grid(True)
plt.savefig(os.path.join(output_path,'volume_per_jam.png'))
plt.show()
