import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Tuple, List, Sequence, Union

from highway_env import utils
from highway_env.vehicle.objects import classify_collision

def get_closed_polygon(position: np.ndarray, heading: float, length: float = 5.0, width: float = 2.0) -> np.ndarray:
    """
    Get polygon corners using highway_env.utils.rect_corners, 
    but repeat the first point at the end to close the loop
    for SAT checks (utils.are_polygons_intersecting expects closed loop).
    """
    # rect_corners returns a list of arrays [p1, p2, p3, p4]
    corners = utils.rect_corners(position, length, width, heading)
    # Stack them into an array
    corners_array = np.array(corners)
    # Append the first point to the end
    return np.vstack([corners_array, corners_array[0:1]])

def analyze_crashes(file_path):
    print(f"Analyzing {file_path} using geometric classification...")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    collision_types = []
    
    total_episodes = 0
    crashed_episodes = 0
    detected_crashes = 0

    # Default Vehicle Params
    LENGTH = 5.0
    WIDTH = 2.0
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                total_episodes += 1
                transitions = record.get("transitions", [])
                
                episode_crash_found = False
                
                for t_idx, t in enumerate(transitions):
                    state = t.get("state")
                    if not state:
                        continue
                    if isinstance(state, list) and len(state) > 0 and isinstance(state[0], list):
                        state = state[0]
                    if len(state) < 10:
                        continue
                    
                    # Consistent Constants
                    n_features = 5
                    n_stack = 5
                    
                    # Dynamic Vehicle Count
                    # total_floats = n_cars * n_features * n_stack
                    # n_cars = total / 25
                    n_cars = int(len(state) // (n_features * n_stack))
                    
                    if n_cars < 2:
                        # Scan skip if only 1 car (can't crash)
                        continue
                    
                    frame_crash_found = False
                    
                    for t_frame in range(n_stack):
                        cars_data = [] # [ [p,x,y,vx,vy], [p,x,y,vx,vy] ]
                        
                        valid_frame = True
                        for c in range(n_cars):
                            c_feats = []
                            for f in range(n_features):
                                # Correct index for [ (T0_C0, T0_C1), (T1_C0, T1_C1)... ] layout
                                idx = (t_frame * n_cars + c) * n_features + f
                                if idx >= len(state):
                                    valid_frame = False
                                    break
                                c_feats.append(state[idx])
                            if not valid_frame:
                                break
                            cars_data.append(c_feats)
                        
                        if not valid_frame:
                            break
                            
                        # Now analyze collision for this frame
                        ego_dat = cars_data[0]
                        npc_dat = cars_data[1]
                        
                        # Check presence (Feature 0)
                        if ego_dat[0] < 0.5 or npc_dat[0] < 0.5:
                            continue
                            
                        ego_pos = np.array([ego_dat[1], ego_dat[2]])
                        npc_pos = np.array([npc_dat[1], npc_dat[2]])
                        
                        ego_vel = np.array([ego_dat[3], ego_dat[4]])
                        npc_vel = np.array([npc_dat[3], npc_dat[4]])
                        
                        ego_heading = np.arctan2(ego_vel[1], ego_vel[0]) if np.linalg.norm(ego_vel) > 0.1 else 0
                        npc_heading = np.arctan2(npc_vel[1], npc_vel[0]) if np.linalg.norm(npc_vel) > 0.1 else 0
                        
                        ego_poly = get_closed_polygon(ego_pos, ego_heading, LENGTH, WIDTH)
                        npc_poly = get_closed_polygon(npc_pos, npc_heading, LENGTH, WIDTH)
                        
                        intersecting, _, mtv = utils.are_polygons_intersecting(
                            ego_poly, npc_poly, np.zeros(2), np.zeros(2)
                        )
                        
                        if intersecting:
                            cl = classify_collision(ego_poly, npc_poly, mtv)
                            
                            c_type = cl.collision_type
                            # Refine side-swipe direction based on ego feature
                            if c_type == "side-swipe":
                                if "left" in cl.ego_feature:
                                    c_type = "side-swipe-left"
                                elif "right" in cl.ego_feature:
                                    c_type = "side-swipe-right"
                                    
                            collision_types.append(c_type)
                            
                            frame_crash_found = True
                            detected_crashes += 1
                            break

                    if frame_crash_found:
                        episode_crash_found = True
                        break 
                
                # Check next_state of the LAST transition if no crash found yet
                # because the crash overlap exists in the terminal state
                if not episode_crash_found and transitions:
                    last_t = transitions[-1]
                    next_state = last_t.get("next_state")
                    if next_state:
                         if isinstance(next_state, list) and len(next_state) > 0 and isinstance(next_state[0], list):
                            next_state = next_state[0]
                         if len(next_state) >= 10:
                             for t_frame in range(n_stack):
                                cars_data = []
                                valid_frame = True
                                for c in range(n_cars):
                                    c_feats = []
                                    for f in range(n_features):
                                        idx = (t_frame * n_cars + c) * n_features + f
                                        if idx >= len(next_state):
                                            valid_frame = False
                                            break
                                        c_feats.append(next_state[idx])
                                    if not valid_frame:
                                        break
                                    cars_data.append(c_feats)
                                
                                if not valid_frame:
                                    break
                                
                                ego_dat = cars_data[0]
                                npc_dat = cars_data[1]
                                
                                # Check presence
                                if ego_dat[0] < 0.5 or npc_dat[0] < 0.5:
                                    continue
                                    
                                ego_pos = np.array([ego_dat[1], ego_dat[2]])
                                npc_pos = np.array([npc_dat[1], npc_dat[2]])
                                
                                ego_vel = np.array([ego_dat[3], ego_dat[4]])
                                npc_vel = np.array([npc_dat[3], npc_dat[4]])
                                
                                ego_heading = np.arctan2(ego_vel[1], ego_vel[0]) if np.linalg.norm(ego_vel) > 0.1 else 0
                                npc_heading = np.arctan2(npc_vel[1], npc_vel[0]) if np.linalg.norm(npc_vel) > 0.1 else 0
                                
                                ego_poly = get_closed_polygon(ego_pos, ego_heading, LENGTH, WIDTH)
                                npc_poly = get_closed_polygon(npc_pos, npc_heading, LENGTH, WIDTH)
                                
                                intersecting, _, mtv = utils.are_polygons_intersecting(
                                    ego_poly, npc_poly, np.zeros(2), np.zeros(2)
                                )
                                
                                if intersecting:
                                    cl = classify_collision(ego_poly, npc_poly, mtv)
                                    c_type = cl.collision_type
                                    if c_type == "side-swipe":
                                        if "left" in cl.ego_feature:
                                            c_type = "side-swipe-left"
                                        elif "right" in cl.ego_feature:
                                            c_type = "side-swipe-right"
                                    
                                    collision_types.append(c_type)
                                    
                                    detected_crashes += 1
                                    episode_crash_found = True
                                    break 
                
                if episode_crash_found:
                    crashed_episodes += 1
                
            except json.JSONDecodeError:
                print("Skipping invalid JSON line")

    print("\n" + "="*40)
    print(f"CRASH: {os.path.basename(file_path)}")
    print("="*40)
    print(f"Total Episodes Processed: {total_episodes}")
    print(f"Episodes with Crashes (detected): {crashed_episodes}")
    if total_episodes > 0:
        print(f"Crash Rate (detected): {crashed_episodes/total_episodes:.2%}")

    if not collision_types:
        print("\nNo crashes detected.")
        return

    print("\nCollision Types Distribution:")
    ct_counts = Counter(collision_types)
    for ctype, count in ct_counts.most_common():
        print(f"  {ctype}: {count}")

    # Plotting - Vertical Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"CRASH: {os.path.basename(file_path)}", fontsize=16)

    if not ct_counts:
        ax.text(0.5, 0.5, "No Crash Data", ha='center', va='center')
    else:
        labels, values = zip(*ct_counts.most_common(10))
        x_pos = np.arange(len(labels))
        # Vertical bar chart: x=types, y=count
        ax.bar(x_pos, values, align='center')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('Collision Type')
        ax.set_ylabel('Count')
        ax.set_title("Distribution of Collision Types")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_img = "crash_analysis_histogram.png"
    plt.savefig(output_img)
    print(f"\nHistogram saved to {output_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze crash data from JSONL file")
    parser.add_argument("file", help="Path to the .jsonl file")
    args = parser.parse_args()
    
    analyze_crashes(args.file)
