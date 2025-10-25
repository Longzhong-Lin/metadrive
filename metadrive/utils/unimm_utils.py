"""
Utility functions for UniMM integration with MetaDrive.
Converts MetaDrive ScenarioDescription to UniMM HeteroData format.

Based on waymo_dataset.py from UniMM-refactor project.
"""

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import HeteroData, Batch

from metadrive.component.traffic_participants.pedestrian import Pedestrian, PedestrianBoundingBox
from metadrive.component.traffic_participants.cyclist import Cyclist, CyclistBoundingBox
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.engine.logger import get_logger
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.type import MetaDriveType

logger = get_logger()


def scenario_to_hetero_data(
    scenario: SD,
    current_step: int,
    scenario_id_to_agent_idx: Dict[str, int],
    dim: int = 2,
    device: str = 'cuda'
) -> HeteroData:
    """
    Convert MetaDrive ScenarioDescription to UniMM HeteroData format.

    Returns data from frame 0 to current_step (inclusive), total of current_step+1 frames.
    The caller (e.g., UniMMTrafficManager) is responsible for slicing the desired
    historical window based on model requirements.

    Args:
        scenario: MetaDrive ScenarioDescription containing tracks and map_features
        current_step: Current simulation step (returns [0, current_step] inclusive)
        scenario_id_to_agent_idx: Mapping from scenario_id to agent index in tensors
        dim: Dimension of position data (2 or 3, default: 2)
        device: PyTorch device ('cuda' or 'cpu')

    Returns:
        HeteroData: PyG HeteroData object with (current_step+1) frames of agent data
    """
    data = {}

    # Extract agent features from scenario tracks
    data['agent'] = _get_agent_features_from_scenario(
        scenario=scenario,
        current_step=current_step,
        scenario_id_to_agent_idx=scenario_id_to_agent_idx,
        dim=dim,
        device=device
    )

    # Extract map features from scenario
    data.update(_get_map_features_from_scenario(
        scenario=scenario,
        dim=dim,
        device=device
    ))

    return Batch.from_data_list([HeteroData(data)])


def _get_agent_features_from_scenario(
    scenario: SD,
    current_step: int,
    scenario_id_to_agent_idx: Dict[str, int],
    dim: int = 2,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Extract agent features from MetaDrive scenario tracks.

    Returns data from frame 0 to current_step (inclusive), total of current_step+1 frames.

    Based on waymo_dataset.py:get_agent_features()

    Returns a dict with keys matching UniMM's expected format:
        - 'num_nodes': int
        - 'valid_mask': (num_agents, current_step+1) bool tensor
        - 'current_valid_mask': (num_agents,) bool tensor
        - 'id': list of agent IDs
        - 'type': (num_agents,) uint8 tensor (agent type)
        - 'position': (num_agents, current_step+1, dim) float tensor
        - 'heading': (num_agents, current_step+1) float tensor
        - 'velocity': (num_agents, current_step+1, dim) float tensor
        - 'length': (num_agents,) float tensor
        - 'width': (num_agents,) float tensor
        - 'height': (num_agents,) float tensor
    """
    tracks = scenario[SD.TRACKS]
    num_agents = len(scenario_id_to_agent_idx)

    # Number of frames to extract: [0, current_step] inclusive
    num_frames = current_step + 1

    # Initialize lists to collect data for each agent
    agent_ids = [None] * num_agents
    valid_masks = []
    positions = []
    headings = []
    velocities = []
    lengths = []
    widths = []
    heights = []
    agent_types = []

    for scenario_id, agent_idx in scenario_id_to_agent_idx.items():
        assert scenario_id in tracks, \
            f"Agent {scenario_id} not found in tracks."

        track = tracks[scenario_id]
        state = track[SD.STATE]

        # Extract data for [0, current_step] frames
        position_full = np.array(state[SD.POSITION])  # (T, 3)
        heading_full = np.array(state[SD.HEADING])    # (T,)
        valid_full = np.array(state['valid'])         # (T,)

        # Slice to [0, current_step] inclusive
        position_hist = position_full[:num_frames, :dim]  # (num_frames, dim)
        heading_hist = heading_full[:num_frames]          # (num_frames,)
        valid_hist = valid_full[:num_frames]              # (num_frames,)

        # Compute velocity
        velocity_full = np.array(state['velocity'])  # (T, 2)
        velocity_hist = np.zeros((num_frames, dim), dtype=np.float32)
        velocity_hist[:, :2] = velocity_full[:num_frames]

        # Extract dimensions (at current step)
        length = state['length'][current_step] if current_step < len(state['length']) else state['length'][-1]
        width = state['width'][current_step] if current_step < len(state['width']) else state['width'][-1]
        height = state['height'][current_step] if current_step < len(state['height']) else state['height'][-1]

        # Map type
        agent_type = _map_metadrive_object_type_to_unimm(track[SD.TYPE])

        # Store data
        agent_ids[agent_idx] = scenario_id
        valid_masks.append(valid_hist > 0)
        positions.append(position_hist)
        headings.append(heading_hist)
        velocities.append(velocity_hist)
        lengths.append(length)
        widths.append(width)
        heights.append(height)
        agent_types.append(agent_type)

    # Convert to tensors
    valid_mask = torch.tensor(np.stack(valid_masks, axis=0), dtype=torch.bool, device=device)  # (num_agents, num_steps)
    current_valid_mask = valid_mask[:, -1].clone()  # (num_agents)

    position = torch.tensor(np.stack(positions, axis=0), dtype=torch.float, device=device)  # (num_agents, num_steps, dim)
    heading = torch.tensor(np.stack(headings, axis=0), dtype=torch.float, device=device)    # (num_agents, num_steps)
    velocity = torch.tensor(np.stack(velocities, axis=0), dtype=torch.float, device=device) # (num_agents, num_steps, dim)

    length = torch.tensor(lengths, dtype=torch.float, device=device)  # (num_agents)
    width = torch.tensor(widths, dtype=torch.float, device=device)    # (num_agents)
    height = torch.tensor(heights, dtype=torch.float, device=device)  # (num_agents)

    agent_type = torch.tensor(agent_types, dtype=torch.uint8, device=device)  # (num_agents)

    # Zero out invalid frames
    position[~valid_mask] = 0.0
    heading[~valid_mask] = 0.0
    velocity[~valid_mask] = 0.0

    return {
        'num_nodes': num_agents,
        'valid_mask': valid_mask,
        'current_valid_mask': current_valid_mask,
        'id': agent_ids,
        'type': agent_type,
        'position': position,
        'heading': heading,
        'velocity': velocity,
        'length': length,
        'width': width,
        'height': height,
    }


def _get_map_features_from_scenario(
    scenario: SD,
    dim: int = 2,
    device: str = 'cuda'
) -> Dict[Union[str, Tuple[str, str, str]], Any]:
    """
    Extract map features from MetaDrive scenario.

    Based on waymo_dataset.py:get_map_features()

    Returns a dict with keys:
        - 'map_polygon': dict
            - 'num_nodes': int
            - 'position': (num_polygons, dim) float tensor
            - 'orientation': (num_polygons,) float tensor
            - 'type': (num_polygons,) long tensor
        - 'map_point': dict
            - 'num_nodes': int
            - 'position': (num_points, dim) float tensor
            - 'orientation': (num_points,) float tensor
            - 'type': (num_points,) long tensor
        - ('map_point', 'to', 'map_polygon'): dict
            - 'edge_index': (2, num_points) long tensor
        - ('map_polygon', 'to', 'map_polygon'): dict
            - 'edge_index': (2, 0) long tensor (empty)
            - 'type': (0,) uint8 tensor (empty)
    """
    map_features = scenario[SD.MAP_FEATURES]

    # Build all_polylines array: [x, y, z, dir_x, dir_y, dir_z, global_type]
    all_polylines = []

    for feature_id, feature in map_features.items():
        feature_type = feature[SD.TYPE]

        if SD.POLYLINE in feature:
            polyline = np.array(feature[SD.POLYLINE], dtype=np.float32)  # (N, 3)
        elif SD.POLYGON in feature:
            polyline = np.array(feature[SD.POLYGON], dtype=np.float32)  # (N, 3)
        elif SD.POSITION in feature:
            polyline = np.array([feature[SD.POSITION]], dtype=np.float32)  # (1, 3)
        else:
            continue

        # Ensure 3D coordinates
        if polyline.shape[1] == 2:
            polyline = np.concatenate([polyline, np.zeros((polyline.shape[0], 1))], axis=-1)

        # Compute direction vectors
        polyline_dir = _get_polyline_dir(polyline)

        # Map type to global type ID
        global_type = _map_metadrive_lane_type_to_unimm(feature_type)
        type_column = np.full((polyline.shape[0], 1), global_type, dtype=np.float32)

        # Concatenate: [x, y, z, dir_x, dir_y, dir_z, global_type]
        polyline_with_features = np.concatenate([polyline, polyline_dir, type_column], axis=-1)
        all_polylines.append(polyline_with_features)

    if len(all_polylines) == 0:
        # Empty map case
        logger.warning("Empty map features in scenario")
        all_polylines = np.zeros((2, 7), dtype=np.float32)
    else:
        all_polylines = np.concatenate(all_polylines, axis=0).astype(np.float32)

    # Generate batch polylines
    batch_polylines, batch_polylines_mask = generate_batch_polylines_from_map(
        all_polylines, num_points_each_polyline=11
    )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)

    batch_polylines_mask = (batch_polylines_mask > 0)
    batch_polylines[~batch_polylines_mask] = 0

    # Downsample map points
    map_downsample_factor = 5
    batch_polylines = batch_polylines[:, ::map_downsample_factor]
    batch_polylines_mask = batch_polylines_mask[:, ::map_downsample_factor]
    map_points = batch_polylines[batch_polylines_mask]  # (num_points, 7)

    num_polygons = batch_polylines.size(0)

    # Compute polygon features
    polygon_position = (batch_polylines[:, :, :dim].sum(dim=1) / torch.clamp_min(
        batch_polylines_mask.sum(dim=1)[:, None].float(), min=1.0)).float()  # (num_polygons, dim)
    polygon_orientation = torch.atan2(
        batch_polylines[:, :, 4].sum(dim=1),
        batch_polylines[:, :, 3].sum(dim=1)
    ).float()  # (num_polygons)

    polygon_type = batch_polylines[:, :, -1][
        torch.arange(num_polygons), torch.argmax(batch_polylines_mask.int(), dim=1)
    ].long()  # (num_polygons)
    polygon_type[polygon_type < 0] = 0
    polygon_type[polygon_type >= 20] = 0

    num_points = map_points.size(0)

    # Compute point features
    point_position = map_points[:, :dim].clone().float()  # (num_points, dim)
    point_orientation = torch.atan2(map_points[:, 4], map_points[:, 3]).float()  # (num_points)

    point_type = map_points[:, -1].clone().long()  # (num_points)
    point_type[point_type < 0] = 0
    point_type[point_type >= 20] = 0

    # Build edge index: point_to_polygon
    polygon_idx, _ = batch_polylines_mask.nonzero(as_tuple=True)
    point_to_polygon_edge_index = torch.stack(
        [torch.arange(num_points, dtype=torch.long), polygon_idx], dim=0
    )  # (2, num_points)

    # Empty polygon_to_polygon edges
    polygon_to_polygon_edge_index = torch.tensor([[], []], dtype=torch.long, device=device)
    polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8, device=device)

    # Move tensors to device
    polygon_position = polygon_position.to(device)
    polygon_orientation = polygon_orientation.to(device)
    polygon_type = polygon_type.to(device)
    point_position = point_position.to(device)
    point_orientation = point_orientation.to(device)
    point_type = point_type.to(device)
    point_to_polygon_edge_index = point_to_polygon_edge_index.to(device)

    map_data = {
        'map_polygon': {},
        'map_point': {},
        ('map_point', 'to', 'map_polygon'): {},
        ('map_polygon', 'to', 'map_polygon'): {},
    }
    map_data['map_polygon']['num_nodes'] = num_polygons
    map_data['map_polygon']['position'] = polygon_position
    map_data['map_polygon']['orientation'] = polygon_orientation
    map_data['map_polygon']['type'] = polygon_type

    map_data['map_point']['num_nodes'] = num_points
    map_data['map_point']['position'] = point_position
    map_data['map_point']['orientation'] = point_orientation
    map_data['map_point']['type'] = point_type

    map_data['map_point', 'to', 'map_polygon']['edge_index'] = point_to_polygon_edge_index
    map_data['map_polygon', 'to', 'map_polygon']['edge_index'] = polygon_to_polygon_edge_index
    map_data['map_polygon', 'to', 'map_polygon']['type'] = polygon_to_polygon_type

    return map_data


def generate_batch_polylines_from_map(
    polylines: np.ndarray,
    point_sampled_interval: int = 1,
    vector_break_dist_thresh: float = 1.0,
    num_points_each_polyline: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate batch polylines from map polylines.

    Based on waymo_dataset.py:generate_batch_polylines_from_map()

    Args:
        polylines: (num_points, 7) array with [x, y, z, dir_x, dir_y, dir_z, global_type]
        point_sampled_interval: Interval for sampling points (default: 1)
        vector_break_dist_thresh: Distance threshold to break polylines (default: 1.0)
        num_points_each_polyline: Number of points per polyline segment (default: 20)

    Returns:
        ret_polylines: (num_polylines, num_points_each_polyline, 7) tensor
        ret_polylines_mask: (num_polylines, num_points_each_polyline) bool tensor
    """
    point_dim = polylines.shape[-1]

    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1)
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[:len(new_polyline)] = new_polyline
        cur_valid_mask[:len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

    ret_polylines = np.stack(ret_polylines, axis=0)
    ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    ret_polylines = torch.from_numpy(ret_polylines)
    ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

    return ret_polylines, ret_polylines_mask


def _get_polyline_dir(polyline: np.ndarray) -> np.ndarray:
    """
    Compute direction vectors for polyline.

    Based on data_preprocess.py:get_polyline_dir()

    Args:
        polyline: (N, 3) array with [x, y, z]

    Returns:
        polyline_dir: (N, 3) array with normalized direction vectors
    """
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def _map_metadrive_object_type_to_unimm(type_str: str) -> int:
    """
    Map MetaDrive object type string to UniMM type index.

    UniMM types:
        0 = TYPE_UNSET
        1 = TYPE_VEHICLE
        2 = TYPE_PEDESTRIAN
        3 = TYPE_CYCLIST
        4 = TYPE_OTHER

    Args:
        type_str: MetaDrive object type string (e.g., 'VEHICLE', 'PEDESTRIAN')

    Returns:
        int: UniMM type index
    """
    if type_str == MetaDriveType.VEHICLE:
        return 1
    elif type_str == MetaDriveType.PEDESTRIAN:
        return 2
    elif type_str == MetaDriveType.CYCLIST:
        return 3
    else:
        return 4  # OTHER


def _map_metadrive_lane_type_to_unimm(lane_type_str: str) -> int:
    """
    Map MetaDrive lane type to UniMM polyline type index.

    UniMM polyline types (from waymo_types.py):
        1: FREEWAY
        2: SURFACE_STREET
        3: BIKE_LANE
        6: ROAD_LINE_BROKEN_SINGLE_WHITE
        7: ROAD_LINE_SOLID_SINGLE_WHITE
        8: ROAD_LINE_SOLID_DOUBLE_WHITE
        9: ROAD_LINE_BROKEN_SINGLE_YELLOW
        10: ROAD_LINE_BROKEN_DOUBLE_YELLOW
        11: ROAD_LINE_SOLID_SINGLE_YELLOW
        12: ROAD_LINE_SOLID_DOUBLE_YELLOW
        13: ROAD_LINE_PASSING_DOUBLE_YELLOW
        15: ROAD_EDGE_BOUNDARY
        16: ROAD_EDGE_MEDIAN
        17: STOP_SIGN
        18: CROSSWALK
        19: SPEED_BUMP
        20: DRIVEWAY

    Args:
        lane_type_str: MetaDrive lane type string

    Returns:
        int: UniMM polyline type index
    """
    # Map MetaDrive types to UniMM types
    type_mapping = {
        # Lane types
        MetaDriveType.LANE_UNKNOWN: -1,
        MetaDriveType.LANE_FREEWAY: 1,
        MetaDriveType.LANE_SURFACE_STREET: 2,
        MetaDriveType.LANE_BIKE_LANE: 3,

        # Road line types
        MetaDriveType.LINE_UNKNOWN: -1,
        MetaDriveType.LINE_BROKEN_SINGLE_YELLOW: 9,
        MetaDriveType.LINE_BROKEN_SINGLE_WHITE: 6,
        MetaDriveType.LINE_SOLID_SINGLE_YELLOW: 11,
        MetaDriveType.LINE_SOLID_SINGLE_WHITE: 7,
        MetaDriveType.LINE_SOLID_DOUBLE_YELLOW: 12,
        MetaDriveType.LINE_SOLID_DOUBLE_WHITE: 8,
        MetaDriveType.LINE_PASSING_DOUBLE_YELLOW: 13,
        MetaDriveType.LINE_BROKEN_DOUBLE_YELLOW: 10,

        # Road edge types
        MetaDriveType.BOUNDARY_LINE: 15,
        MetaDriveType.BOUNDARY_MEDIAN: 16,

        # Other types
        MetaDriveType.STOP_SIGN: 17,
        MetaDriveType.CROSSWALK: 18,
        MetaDriveType.SPEED_BUMP: 19,
        MetaDriveType.DRIVEWAY: 20,
    }

    return type_mapping.get(lane_type_str, 2)  # Default to SURFACE_STREET
