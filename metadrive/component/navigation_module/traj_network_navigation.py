"""
TrajNetworkNavigation: Trajectory-based navigation compatible with ExpertPolicy

This navigation module combines the advantages of TrajectoryNavigation and NodeNetworkNavigation:
- Uses reference trajectory directly (like TrajectoryNavigation) - no dependency on lane graph
- Provides 10-dim navigation observation (like NodeNetworkNavigation) - compatible with ExpertPolicy
- Estimates curvature from trajectory geometry instead of relying on CircularLane properties

Designed for ScenarioEnv with EdgeRoadNetwork where strict graph structure is not available.
"""

import numpy as np
from collections import deque
from metadrive.component.navigation_module.base_navigation import BaseNavigation
from metadrive.component.pg_space import Parameter, BlockParameterSpace
from metadrive.utils import clip, norm
from metadrive.utils.math import panda_vector, wrap_to_pi


class TrajNetworkNavigation(BaseNavigation):
    """
    Navigation module that generates NodeNetworkNavigation-compatible observations
    from reference trajectory data.

    Key features:
    - 10-dim navigation info (2 waypoints × 5 dims): compatible with ExpertPolicy
    - Direct trajectory sampling with configurable lookahead distance
    - Curvature estimation: computed from trajectory geometry
    """

    # Match NodeNetworkNavigation dimensions
    NUM_WAY_POINT = 2
    CHECK_POINT_INFO_DIM = 5
    NAVI_POINT_DIST = 50  # meters, used to normalize checkpoint position

    # Lookahead distances (meters) - can be overridden via vehicle_config
    LOOKAHEAD_DISTANCE_1 = 20.0  # First checkpoint lookahead distance
    LOOKAHEAD_DISTANCE_2 = 40.0  # Second checkpoint lookahead distance
    CURVATURE_LOOKAHEAD = 30.0  # Distance for curvature estimation

    def __init__(
        self,
        show_navi_mark: bool = False,
        show_dest_mark=False,
        show_line_to_dest=False,
        panda_color=None,
        name=None,
        vehicle_config=None
    ):
        super(TrajNetworkNavigation, self).__init__(
            show_navi_mark=show_navi_mark,
            show_dest_mark=show_dest_mark,
            show_line_to_dest=show_line_to_dest,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )

        # Override lookahead distances from vehicle_config if provided
        if vehicle_config is not None:
            self.LOOKAHEAD_DISTANCE_1 = vehicle_config.get('lookahead_distance_1', self.LOOKAHEAD_DISTANCE_1)
            self.LOOKAHEAD_DISTANCE_2 = vehicle_config.get('lookahead_distance_2', self.LOOKAHEAD_DISTANCE_2)
            self.CURVATURE_LOOKAHEAD = vehicle_config.get('curvature_lookahead', self.CURVATURE_LOOKAHEAD)

        # Tracking for route completion and localization
        self._route_completion = 0.0
        self.last_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_current_heading_theta_at_long = deque([0.0, 0.0], maxlen=2)

        # For compatibility with other code
        self.next_ref_lanes = None
        self.current_ref_lanes = None
        self.final_lane = None

    def reset(self, vehicle):
        """Reset navigation for a new episode."""
        # Get reference trajectory from map manager
        ref_traj = self.reference_trajectory
        if ref_traj is None:
            raise ValueError("No reference trajectory available for navigation")

        # Use trajectory as current lane for BaseNavigation
        super(TrajNetworkNavigation, self).reset(current_lane=ref_traj)

        # Set up references
        self.set_route()
        self.current_ref_lanes = [ref_traj]
        self.final_lane = ref_traj

    @property
    def reference_trajectory(self):
        """Get the reference trajectory from map manager."""
        return self.engine.map_manager.current_sdc_route

    def set_route(self):
        """Initialize route markers."""
        self._navi_info.fill(0.0)
        self.next_ref_lanes = None

        # Set destination marker if visualization is enabled
        if self._dest_node_path is not None:
            dest_point = self.reference_trajectory.end
            self._dest_node_path.setPos(panda_vector(dest_point[0], dest_point[1], self.MARK_HEIGHT))

    def update_localization(self, ego_vehicle):
        """
        Update navigation information every step.

        This is the core function that computes the 10-dim navigation observation.
        Directly samples lookahead points from the reference trajectory.
        """
        if self.reference_trajectory is None:
            return

        ref_traj = self.reference_trajectory

        # Get vehicle's position on trajectory
        long, lat = ref_traj.local_coordinates(ego_vehicle.position)
        heading_theta_at_long = ref_traj.heading_theta_at(long)

        # Update tracking deques
        self.last_current_long.append(long)
        self.last_current_lat.append(lat)
        self.last_current_heading_theta_at_long.append(heading_theta_at_long)

        # Update route completion
        self._route_completion = clip(long / ref_traj.length, 0.0, 1.0)

        # Directly sample lookahead points from trajectory
        checkpoint_long_1 = min(long + self.LOOKAHEAD_DISTANCE_1, ref_traj.length)
        checkpoint_long_2 = min(long + self.LOOKAHEAD_DISTANCE_2, ref_traj.length)

        checkpoint_1 = ref_traj.position(checkpoint_long_1, 0)
        checkpoint_2 = ref_traj.position(checkpoint_long_2, 0)

        # Compute 10-dim navigation info (2 waypoints × 5 dims each)
        self._navi_info.fill(0.0)
        half = self.CHECK_POINT_INFO_DIM

        # First checkpoint info (dims 0-4)
        self._navi_info[:half], heading_1, pos_1 = self._get_info_for_checkpoint(
            checkpoint_position=checkpoint_1,
            checkpoint_long=checkpoint_long_1,
            ego_vehicle=ego_vehicle,
            is_current=True
        )

        # Second checkpoint info (dims 5-9)
        self._navi_info[half:], heading_2, pos_2 = self._get_info_for_checkpoint(
            checkpoint_position=checkpoint_2,
            checkpoint_long=checkpoint_long_2,
            ego_vehicle=ego_vehicle,
            is_current=False
        )

        # Update visualization if enabled
        if self._show_navi_info:
            self._goal_node_path.setPos(panda_vector(pos_1[0], pos_1[1], self.MARK_HEIGHT))
            self._goal_node_path.setH(self._goal_node_path.getH() + 3)

            self._goal_node_path2.setPos(panda_vector(pos_2[0], pos_2[1], self.MARK_HEIGHT))
            self._goal_node_path2.setH(self._goal_node_path2.getH() + 3)

            self.navi_arrow_dir = [heading_1, heading_2]

            dest_pos = self._dest_node_path.getPos()
            self._draw_line_to_dest(
                start_position=ego_vehicle.position,
                end_position=(dest_pos[0], dest_pos[1])
            )

            navi_pos = self._goal_node_path.getPos()
            next_navi_pos = self._goal_node_path2.getPos()
            self._draw_line_to_navi(
                start_position=ego_vehicle.position,
                end_position=(navi_pos[0], navi_pos[1]),
                next_checkpoint=(next_navi_pos[0], next_navi_pos[1])
            )

    def _get_info_for_checkpoint(self, checkpoint_position, checkpoint_long, ego_vehicle, is_current):
        """
        Compute 5-dim navigation information for a checkpoint.

        This matches NodeNetworkNavigation's format:
        - Dim 1: Checkpoint position in heading direction [0, 1]
        - Dim 2: Checkpoint position in lateral direction [0, 1]
        - Dim 3: Path curvature (estimated from trajectory) [0, 1]
        - Dim 4: Curvature direction [0, 1]
        - Dim 5: Angular change [0, 1]

        Args:
            checkpoint_position: 2D position of the checkpoint
            checkpoint_long: Longitudinal position on trajectory
            ego_vehicle: The vehicle object
            is_current: Whether this is the immediate next checkpoint

        Returns:
            Tuple of (5-dim numpy array, heading_theta, checkpoint_position)
        """
        ref_traj = self.reference_trajectory
        navi_info = []

        # === Dim 1 & 2: Relative position in vehicle's coordinate frame ===
        dir_vec = checkpoint_position - ego_vehicle.position
        dir_norm = norm(dir_vec[0], dir_vec[1])

        # Clip to maximum navigation distance
        if dir_norm > self.NAVI_POINT_DIST:
            dir_vec = dir_vec / dir_norm * self.NAVI_POINT_DIST

        # Convert to vehicle's local coordinates (+x = heading, +y = right)
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(dir_vec, 0.0)

        # Normalize to [0, 1] range
        navi_info.append(clip((ckpt_in_heading / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))
        navi_info.append(clip((ckpt_in_rhs / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        # === Get heading at checkpoint ===
        if is_current:
            # Use heading at vehicle's current position for immediate checkpoint
            heading_theta = ref_traj.heading_theta_at(
                ref_traj.local_coordinates(ego_vehicle.position)[0]
            )
        else:
            # Use heading at checkpoint position for next checkpoint
            heading_theta = ref_traj.heading_theta_at(
                min(checkpoint_long, ref_traj.length)
            )

        # === Dim 3, 4, 5: Curvature information ===
        # Estimate curvature by analyzing trajectory geometry
        curvature_info = self._estimate_curvature(checkpoint_long, ref_traj)
        navi_info.extend(curvature_info)

        return np.array(navi_info, dtype=np.float32), heading_theta, checkpoint_position

    def _estimate_curvature(self, longitudinal, trajectory):
        """
        Estimate curvature at a point on the trajectory.

        We approximate curvature by looking at the angular change over a lookahead distance.
        Returns 3 values matching NodeNetworkNavigation format:
        - bend_radius: normalized radius of curvature [0, 1]
        - direction: bending direction [-1 for clockwise, +1 for counter-clockwise]
        - angle: total angular change over the sample distance

        Args:
            longitudinal: Position along trajectory
            trajectory: The reference trajectory (InterpolatingLine)

        Returns:
            List of 3 float values [bend_radius, direction, angle]
        """
        # Use configurable lookahead distance for curvature estimation
        sample_len = self.CURVATURE_LOOKAHEAD

        # Ensure we stay within trajectory bounds
        long_start = max(0, longitudinal)
        long_end = min(trajectory.length, longitudinal + sample_len)

        # Get headings at start and end
        try:
            heading_start = trajectory.heading_theta_at(long_start)
            heading_end = trajectory.heading_theta_at(long_end)
        except (AttributeError, IndexError):
            # If trajectory doesn't support heading queries, assume straight
            return [0.0, 0.5, 0.5]  # straight road

        # Calculate angular change
        total_angle_change = wrap_to_pi(heading_end - heading_start)

        # If nearly straight, return zero curvature
        if abs(total_angle_change) < 0.01:  # ~0.5 degrees
            return [0.0, 0.5, 0.5]

        # Estimate radius of curvature
        # For a circular arc: radius = arc_length / angle
        actual_distance = long_end - long_start
        if actual_distance < 0.1:  # Avoid division by near-zero
            return [0.0, 0.5, 0.5]

        # Radius of curvature
        radius = abs(actual_distance / total_angle_change) if abs(total_angle_change) > 1e-6 else 1e6

        # Normalize radius to [0, 1] range
        # Use typical road curvature bounds from BlockParameterSpace
        max_radius = BlockParameterSpace.CURVE.get(Parameter.radius, type('obj', (), {'max': 100})).max
        normalized_radius = clip(radius / max_radius, 0.0, 1.0)

        # Determine turn direction
        # Positive angle change = left turn (counter-clockwise)
        # Negative angle change = right turn (clockwise)
        direction = 1.0 if total_angle_change > 0 else -1.0

        # Normalize direction to [0, 1] where 0 = clockwise, 1 = counter-clockwise
        normalized_direction = clip((direction + 1) / 2, 0.0, 1.0)

        # Normalize angle to [0, 1] range
        # Use typical maximum curve angle from BlockParameterSpace
        max_angle = BlockParameterSpace.CURVE.get(Parameter.angle, type('obj', (), {'max': np.pi/2})).max
        normalized_angle = clip(
            (abs(total_angle_change) / max_angle + 1) / 2,
            0.0,
            1.0
        )

        return [normalized_radius, normalized_direction, normalized_angle]

    def get_current_lateral_range(self, current_position, engine) -> float:
        """Return the lateral range (for compatibility)."""
        return self.current_lane.width * 2 if self.current_lane is not None else 7.0

    def get_current_lane_width(self) -> float:
        """Return current lane width (for compatibility)."""
        return self.current_lane.width if self.current_lane is not None else 3.5

    def get_current_lane_num(self) -> float:
        """Return number of lanes (for compatibility)."""
        return 1.0

    # === Properties for ScenarioEnv compatibility ===

    @property
    def route_completion(self):
        """Route completion progress [0, 1]."""
        return self._route_completion

    @property
    def last_longitude(self):
        """Last longitudinal position on trajectory."""
        return self.last_current_long[0]

    @property
    def current_longitude(self):
        """Current longitudinal position on trajectory."""
        return self.last_current_long[1]

    @property
    def last_lateral(self):
        """Last lateral position relative to trajectory."""
        return self.last_current_lat[0]

    @property
    def current_lateral(self):
        """Current lateral position relative to trajectory."""
        return self.last_current_lat[1]

    @property
    def last_heading_theta_at_long(self):
        """Last heading theta at longitudinal position."""
        return self.last_current_heading_theta_at_long[0]

    @property
    def current_heading_theta_at_long(self):
        """Current heading theta at longitudinal position."""
        return self.last_current_heading_theta_at_long[1]

    @classmethod
    def get_navigation_info_dim(cls):
        """Return navigation observation dimension: 10 (2 waypoints × 5 dims)."""
        return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM

    def destroy(self):
        """Clean up resources."""
        self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        super(TrajNetworkNavigation, self).destroy()

    def before_reset(self):
        """Called before reset."""
        self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
