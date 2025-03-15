'''
Some of this script is adapted from the TUM RGB-D dataset tools:
Jürgen Sturm et al. (2012) "A Benchmark for the Evaluation of RGB-D SLAM Systems"
Source: https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py
'''
import numpy as np
import matplotlib.pyplot as plt

def inspect_depth(depth_meters):
    # Convert depth values to meters
    # depth_meters = depth.astype(np.float32)
    depth_meters

    # Remove zero values (missing depth) for histogram
    depth_values = depth_meters[depth_meters > 0]

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Histogram of depth values
    axes[0].hist(depth_values.flatten(), bins=100, color='blue', alpha=0.7)
    axes[0].set_xlabel("Depth Value (meters)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram of Depth Values")
    axes[0].grid(True)

    # Grayscale depth map
    im1 = axes[1].imshow(depth_meters, cmap='gray', interpolation='nearest')
    axes[1].set_title("Depth Map in Grayscale")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (meters)")

    # Jet colormap depth map
    im2 = axes[2].imshow(depth_meters, cmap='jet', interpolation='nearest')
    axes[2].set_title("Colored Depth Map (Jet Colormap)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Depth (meters)")

    # Adjust layout
    plt.tight_layout()
    plt.show()

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list, offset=0.0, max_difference=0.02):
    """
    Associate two dictionaries of (timestamp, data). 
    Finds the closest match for every timestamp in the first list.
    
    Input:
    - first_list: Dictionary of (timestamp, data) pairs
    - second_list: Dictionary of (timestamp, data) pairs
    - offset: Time offset added to second_list timestamps
    - max_difference: Maximum allowed difference for a valid match

    Output:
    - matches: List of matched tuples ((timestamp1, data1), (timestamp2, data2))
    """
    first_keys = list(first_list.keys())  # Convert to a mutable list
    second_keys = list(second_list.keys())  # Convert to a mutable list

    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in first_keys
        for b in second_keys
        if abs(a - (b + offset)) < max_difference
    ]
    
    potential_matches.sort()  # Sort by closest timestamp match
    matches = []

    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)  # Now works correctly
            second_keys.remove(b)  # Now works correctly
            matches.append((a, b))

    matches.sort()
    return matches

def associate_time_stamps(first_file, second_file, first_only=False, offset=0.0, max_difference=0.02):
    """
    Associates timestamps from two files and returns the matched pairs.

    Input:
    - first_file: Path to the first text file (e.g., rgb.txt).
    - second_file: Path to the second text file (e.g., depth.txt).
    - first_only: If True, only returns matched entries from the first file.
    - offset: Time offset added to the timestamps of the second file.
    - max_difference: Maximum allowed time difference for matching entries.

    Output:
    - matches: List of matched tuples. 
      If first_only=True, returns [(timestamp1, data1), ...]
      If first_only=False, returns [(timestamp1, data1, timestamp2, data2), ...]
    """
    first_list = read_file_list(first_file)
    second_list = read_file_list(second_file)

    matches = associate(first_list, second_list, offset, max_difference)
    
    # Return matches in the appropriate format
    if first_only:
        return [(a, first_list[a]) for a, _ in matches]
    else:
        return [(a, first_list[a], b, second_list[b]) for a, b in matches]
    
def evaluate_matched_pairs(matched_pairs):
    print(len(matched_pairs))

    # Extract timestamp differences
    time_diffs = [abs(rgb_ts - depth_ts) for rgb_ts, _, depth_ts, _ in matched_pairs]

    # Print basic statistics
    print(f"Mean timestamp difference: {np.mean(time_diffs):.6f} sec")
    print(f"Max timestamp difference: {np.max(time_diffs):.6f} sec")
    print(f"Min timestamp difference: {np.min(time_diffs):.6f} sec")
    print(f"95th percentile difference: {np.percentile(time_diffs, 95):.6f} sec")

    # Plot histogram of timestamp differences
    plt.figure(figsize=(8, 5))
    plt.hist(time_diffs, bins=50, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel("Timestamp Difference (seconds)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Timestamp Differences Between RGB and Depth")
    plt.grid(True)
    plt.show()