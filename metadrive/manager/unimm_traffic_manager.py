"""
UniMM Traffic Manager - Reactive traffic flow using UniMM model
"""

import copy
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from metadrive.component.traffic_participants.pedestrian import Pedestrian, PedestrianBoundingBox
from metadrive.component.traffic_participants.cyclist import Cyclist, CyclistBoundingBox
from metadrive.constants import DEFAULT_AGENT
from metadrive.engine.logger import get_logger
from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager, get_vehicle_type
from metadrive.scenario.parse_object_state import parse_object_state
from metadrive.type import MetaDriveType
from metadrive.utils.unimm_utils import scenario_to_hetero_data

from sim_agents.models.unimm import UniMM
from sim_agents.utils import wrap_angle

logger = get_logger()


class UniMMTrafficManager(ScenarioTrafficManager):
    """Traffic Manager using UniMM model for reactive traffic prediction."""

    def __init__(self):
        super(UniMMTrafficManager, self).__init__()

        # UniMM model and state
        self.model = None
        self.device = None

        # Persistent state for reactive inference
        self.sim_data = None  # HeteroData with sliding window
        self.map_enc = None  # Map encoding (computed once)
        self.temporal_cache = None  # Transformer temporal cache

        # Prediction cache
        self.predictions = None  # Current multi-frame predictions
        self.prediction_start_frame = 0

        # Agent mapping: scenario_id <-> agent_index in UniMM tensors
        self._scenario_id_to_agent_idx = {}
        self._agent_idx_to_scenario_id = {}

        # Step counter (used to determine if step==0 for key_frames calculation)
        self.model_step_count = 0

    def before_reset(self):
        super(UniMMTrafficManager, self).before_reset()

        # Load UniMM model if not loaded
        if self.model is None:
            checkpoint_path = self.engine.global_config.get("unimm_checkpoint")
            self.device = self.engine.global_config.get("unimm_device", 'cuda' if torch.cuda.is_available() else 'cpu')
            self.load_unimm_model(checkpoint_path)

    def after_reset(self):
        """Initialize traffic vehicles and perform first prediction"""
        # Reset mappings
        self._scenario_id_to_obj_id = {}
        self._obj_id_to_scenario_id = {}
        self._scenario_id_to_agent_idx = {}
        self._agent_idx_to_scenario_id = {}

        self._static_car_id = set()
        self._moving_car_id = set()
        self._noise_object_id = set()
        self._non_noise_object_id = set()

        # Reset prediction records
        self.predictions = None
        self.prediction_start_frame = 0
        self.model_step_count = 0

        # Spawn all traffic participants (without policies - we'll control them directly)
        for scenario_id, track in self.current_traffic_data.items():
            if scenario_id == self.sdc_scenario_id:
                continue
            if track["type"] == MetaDriveType.VEHICLE:
                self.spawn_vehicle(scenario_id, track)
            elif track["type"] == MetaDriveType.CYCLIST:
                self.spawn_cyclist(scenario_id, track)
            elif track["type"] == MetaDriveType.PEDESTRIAN:
                self.spawn_pedestrian(scenario_id, track)
            else:
                logger.warning("Do not support {}".format(track["type"]))

        # Build agent index mapping
        self._build_agent_index_mapping()

        # Get sim_data from scenario
        self.sim_data = scenario_to_hetero_data(
            scenario=self.engine.data_manager.current_scenario,
            current_step=self.episode_step,
            scenario_id_to_agent_idx=self._scenario_id_to_agent_idx,
            dim=2,  # UniMM uses 2D coordinates by default
            device=self.device if self.device else 'cpu'
        )
        assert self.sim_data['agent']['current_valid_mask'].all()
        
        # Initialize UniMM inference
        self.map_enc = self.model.encoder.map_encoder(self.sim_data)
        self.temporal_cache = None

    def after_step(self, *args, **kwargs):
        """Called every frame at 10Hz"""

        frame_offset = self.episode_step - self.prediction_start_frame
        
        # Update sim_data with current simulation state
        if self.predictions is not None:
            self._update_sim_data_with_current_world_state(frame_offset)
        
        # Update UniMM predictions and apply to simulation
        if self.predictions is None or frame_offset >= self.model.num_execution_steps:
            self._update_predictions()
            self._apply_predictions(0)
        else:
            self._apply_predictions(frame_offset)

        # TODO: Handle vehicle spawning/despawning during episode

        return dict(default_agent=dict(replay_done=False))

    def load_unimm_model(self, checkpoint_path):
        """Load UniMM model from checkpoint."""
        assert checkpoint_path is not None, "Missing UniMM checkpoint path."
        logger.info(f"Loading UniMM model from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Extract hyperparameters
        hparams = checkpoint['hyper_parameters']
        # Create model
        self.model = UniMM(**hparams)
        # Load weights
        self.model.load_state_dict(checkpoint['state_dict'])
        # Move to device and set to eval mode
        self.model.to(self.device).eval()

        # Check model validity
        assert self.model.key_frame_interval <= self.model.num_execution_steps
        assert self.model.num_execution_steps <= self.model.num_prediction_steps
        assert self.model.input_dim == self.model.output_dim == 2

        logger.info(f"UniMM model loaded successfully on {self.device}")

    def spawn_vehicle(self, v_id, track):
        """Spawn vehicle without policy (UniMM will control it directly)"""
        state = parse_object_state(track, self.episode_step, include_z_position=False)
        use_bounding_box = (
            self.engine.global_config["vehicle_config"]["vehicle_model"] == "varying_dynamics_bounding_box"
        )

        # for each vehicle, we would like to know if it is static
        if v_id not in self._static_car_id and v_id not in self._moving_car_id:
            valid_points = track["state"]["position"][np.where(track["state"]["valid"])]
            moving = np.max(np.std(valid_points, axis=0)[:2]) > self.STATIC_THRESHOLD
            set_to_add = self._moving_car_id if moving else self._static_car_id
            set_to_add.add(v_id)

        # don't create in these two conditions
        if not state["valid"] or (self.engine.global_config["no_static_vehicles"] and v_id in self._static_car_id):
            return

        # if collision don't generate, unless ego car is in replay mode
        ego_pos = self.ego_vehicle.position
        heading_dist, side_dist = self.ego_vehicle.convert_to_local_coordinates(state["position"][:2], ego_pos)
        if not self.is_ego_vehicle_replay and self._filter_overlapping_car and \
                abs(heading_dist) < self.GENERATION_FORWARD_CONSTRAINT and \
                abs(side_dist) < self.GENERATION_SIDE_CONSTRAINT:
            return

        # create vehicle
        if state["vehicle_class"] and not use_bounding_box:
            vehicle_class = state["vehicle_class"]
        else:
            vehicle_class = get_vehicle_type(
                float(state["length"]), self.need_default_vehicle, use_bounding_box=use_bounding_box
            )
        # print("vehicle_class: ", vehicle_class)
        obj_name = v_id if self.engine.global_config["force_reuse_object_name"] else None
        v_cfg = copy.copy(self._traffic_v_config)

        v_cfg["width"] = state["width"]
        v_cfg["length"] = state["length"]
        v_cfg["height"] = state["height"]
        if use_bounding_box:
            v_cfg["scale"] = (
                v_cfg["width"] / vehicle_class.DEFAULT_WIDTH, v_cfg["length"] / vehicle_class.DEFAULT_LENGTH,
                v_cfg["height"] / vehicle_class.DEFAULT_HEIGHT
            )

        if self.engine.global_config["top_down_show_real_size"]:
            v_cfg["top_down_length"] = track["state"]["length"][self.episode_step]
            v_cfg["top_down_width"] = track["state"]["width"][self.episode_step]
            if v_cfg["top_down_length"] < 1 or v_cfg["top_down_width"] < 0.5:
                logger.warning(
                    "Scenario ID: {}. The top_down size of vehicle {} is weird: "
                    "{}".format(self.engine.current_seed, v_id, [v_cfg["length"], v_cfg["width"]])
                )

        position = list(state["position"])

        # Add z to make it stick to the ground:
        assert len(position) == 2
        if use_bounding_box:
            position.append(state['height'] / 2)

        v = self.spawn_object(
            vehicle_class, position=position, heading=state["heading"], vehicle_config=v_cfg, name=obj_name
        )
        self._scenario_id_to_obj_id[v_id] = v.name
        self._obj_id_to_scenario_id[v.name] = v_id

        # NOTE: No policy is added! UniMM will control the vehicle directly via set_position/heading/velocity

    def spawn_pedestrian(self, scenario_id, track):
        """Spawn pedestrian without policy (UniMM will control it directly)"""
        state = parse_object_state(track, self.episode_step, include_z_position=False)
        if not state["valid"]:
            return
        obj_name = scenario_id if self.engine.global_config["force_reuse_object_name"] else None
        if self.global_config["use_bounding_box"]:
            cls = PedestrianBoundingBox
            force_spawn = True
        else:
            cls = Pedestrian
            force_spawn = False

        position = list(state["position"])
        obj = self.spawn_object(
            cls,
            name=obj_name,
            position=position,
            heading_theta=state["heading"],
            width=state["width"],
            length=state["length"],
            height=state["height"],
            force_spawn=force_spawn
        )
        self._scenario_id_to_obj_id[scenario_id] = obj.name
        self._obj_id_to_scenario_id[obj.name] = scenario_id

        # NOTE: No policy is added! UniMM will control the pedestrian directly

    def spawn_cyclist(self, scenario_id, track):
        """Spawn cyclist without policy (UniMM will control it directly)"""
        state = parse_object_state(track, self.episode_step, include_z_position=False)
        if not state["valid"]:
            return
        obj_name = scenario_id if self.engine.global_config["force_reuse_object_name"] else None
        if self.global_config["use_bounding_box"]:
            cls = CyclistBoundingBox
            force_spawn = True
        else:
            cls = Cyclist
            force_spawn = False

        position = list(state["position"])
        obj = self.spawn_object(
            cls,
            name=obj_name,
            position=position,
            heading_theta=state["heading"],
            width=state["width"],
            length=state["length"],
            height=state["height"],
            force_spawn=force_spawn
        )
        self._scenario_id_to_obj_id[scenario_id] = obj.name
        self._obj_id_to_scenario_id[obj.name] = scenario_id

        # NOTE: No policy is added! UniMM will control the cyclist directly

    def _build_agent_index_mapping(self):
        """Build mapping from scenario_id to agent index in UniMM tensors"""
        # TODO: Define ordering (e.g., alphabetical by scenario_id)
        # For now, use sorted order
        all_agent_ids = sorted([sid for sid in self._scenario_id_to_obj_id.keys()])

        # Add ego vehicle first (ego should be included in UniMM input but not controlled)
        agent_list = [self.sdc_scenario_id] + all_agent_ids

        for idx, scenario_id in enumerate(agent_list):
            self._scenario_id_to_agent_idx[scenario_id] = idx
            self._agent_idx_to_scenario_id[idx] = scenario_id

        logger.info(f"Built agent mapping: {len(agent_list)} agents (1 ego + {len(all_agent_ids)} traffic)")

    def _update_sim_data_with_current_world_state(self, frame_index):
        """Update a specific frame of sim_data with current world state (including ego)."""
        
        for scenario_id, agent_idx in self._scenario_id_to_agent_idx.items():
            # Find vehicle by scenario_id
            if scenario_id == self.sdc_scenario_id:
                vehicle = self.ego_vehicle
            elif scenario_id in self._scenario_id_to_obj_id:
                obj_id = self._scenario_id_to_obj_id[scenario_id]
                if obj_id not in self.spawned_objects:
                    continue
                vehicle = self.spawned_objects[obj_id]
            else:
                continue

            # Update specified frame of sim_data
            pos = vehicle.position  # (x, y)
            self.sim_data['agent']['position'][agent_idx, frame_index] = torch.tensor(pos, dtype=torch.float, device=self.device)

            heading = vehicle.heading_theta
            self.sim_data['agent']['heading'][agent_idx, frame_index] = torch.tensor(heading, dtype=torch.float, device=self.device)

            vel = vehicle.velocity  # (vx, vy)
            self.sim_data['agent']['velocity'][agent_idx, frame_index] = torch.tensor(vel, dtype=torch.float, device=self.device)

    @torch.no_grad()
    def _update_predictions(self):
        """Call UniMM to generate next `num_execution_steps` frames of predictions"""
        # This replicates the core logic of inference() rollout loop

        # 1. Calculate key_frames
        key_frames = self._calculate_key_frames()

        # 2. Encode agents (using temporal_cache)
        agent_enc = self.model.encoder.agent_encoder.inference(
            self.sim_data, self.map_enc,
            key_frames=key_frames,
            temporal_cache=self.temporal_cache,
        )
        self.temporal_cache = agent_enc['temporal_cache']

        # 3. Predict trajectories
        x_a_current = agent_enc['x_a'][:, -1]

        if self.model.anchor_free:
            # Anchor-free: directly predict trajectories
            pred = self.model.decoder(x_a_current.unsqueeze(1))
            score = pred['pi'].squeeze(1)
            loc_refine_pos = pred['loc_refine_pos'][..., :self.model.output_dim].squeeze(1)
            loc_refine_head = pred['loc_refine_head'][..., -1].squeeze(1)
        else:
            # Anchor-based: load anchors and score them
            propose_traj = self.model.anchor_trajs[
                self.sim_data['agent']['type'].long(), :, :self.model.num_prediction_steps
            ]  # (num_agents, num_modes, num_prediction_steps, traj_dim)

            score = self.model.scorer(x_a_current)

        # 4. Sample mode
        sample_pi = F.softmax(score, dim=-1)
        sample_mode = torch.multinomial(sample_pi, 1, replacement=True).squeeze(-1)

        # 5. Extract predicted trajectory for sampled mode
        if self.model.anchor_free:
            pred_position = torch.gather(
                loc_refine_pos, 1,
                sample_mode[:, None, None, None].expand(
                    -1, -1, loc_refine_pos.size(-2), loc_refine_pos.size(-1)
                )
            ).squeeze(1)  # (num_agents, num_prediction_steps, 2)

            pred_heading = torch.gather(
                loc_refine_head, 1,
                sample_mode[:, None, None].expand(-1, -1, loc_refine_head.size(-1))
            ).squeeze(1)  # (num_agents, num_prediction_steps)
        else:
            # Anchor-based: gather sampled anchor trajectories
            pred_traj = torch.gather(
                propose_traj, 1,
                sample_mode[:, None, None, None].expand(
                    -1, -1, propose_traj.size(-2), propose_traj.size(-1)
                )
            ).squeeze(1)  # (num_agents, num_prediction_steps, traj_dim)

            pred_position = pred_traj[..., :self.model.output_dim]
            pred_heading = pred_traj[..., -1]

        # 6. Refine trajectory if decoder exists (for anchor-based models)
        if not self.model.anchor_free and self.model.reg_decoder is not None:
            pred = self.model.decoder(x_a_current.unsqueeze(1), pred_traj.unsqueeze(1))
            pred_position = pred['loc_refine_pos'][..., :self.model.output_dim].squeeze(1)
            pred_heading = pred['loc_refine_head'][..., -1].squeeze(1)

        # 7. Convert to global coordinates
        origin_global = self.sim_data['agent']['position'][:, -1]
        theta_global = self.sim_data['agent']['heading'][:, -1]
        cos, sin = theta_global.cos(), theta_global.sin()

        rot_mat = torch.zeros(theta_global.shape[0], 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos

        sample_position = torch.matmul(
            pred_position[:, :, :2], rot_mat
        ) + origin_global[:, :2].unsqueeze(1)

        sample_heading = wrap_angle(pred_heading + theta_global.unsqueeze(1))

        sample_velocity = (
            sample_position - torch.cat([origin_global[:, None, :2], sample_position[:, :-1]], dim=1)
        ) * 10  # 10 Hz

        # 8. Update sim_data sliding window (keep last 1 frame + new `num_execution_steps` frames)
        num_exec = self.model.num_execution_steps
        update_mask = torch.ones(
            self.sim_data['agent']['num_nodes'], num_exec, device=self.device, dtype=torch.bool
        )
        update_position = sample_position[:, :num_exec]
        update_heading = sample_heading[:, :num_exec]
        update_velocity = sample_velocity[:, :num_exec]

        self.sim_data['agent']['valid_mask'] = torch.cat([
            self.sim_data['agent']['valid_mask'][:, -1:], update_mask
        ], dim=1)
        self.sim_data['agent']['position'] = torch.cat([
            self.sim_data['agent']['position'][:, -1:, :self.model.input_dim], update_position
        ], dim=1)
        self.sim_data['agent']['heading'] = torch.cat([
            self.sim_data['agent']['heading'][:, -1:], update_heading
        ], dim=1)
        self.sim_data['agent']['velocity'] = torch.cat([
            self.sim_data['agent']['velocity'][:, -1:, :self.model.input_dim], update_velocity
        ], dim=1)

        # 9. Cache predictions for next `num_execution_steps` frames
        self.predictions = {
            'positions': update_position,
            'headings': update_heading,
            'velocities': update_velocity,
        }
        self.prediction_start_frame = self.episode_step

        self.model_step_count += 1

    def _calculate_key_frames(self):
        """Calculate key frames for UniMM model"""
        
        if self.model_step_count == 0:  # First step
            num_history_frames = self.sim_data['agent']['valid_mask'].shape[1]
            start_frame = (num_history_frames % self.model.key_frame_interval) - 1
            if start_frame < 0:
                start_frame += self.model.key_frame_interval
            key_frames = torch.arange(
                start_frame, num_history_frames, self.model.key_frame_interval, device=self.device
            )
            # add 0 to key_frames
            key_frames = torch.cat([torch.tensor([0], device=self.device), key_frames])
            key_frames = key_frames.sort().values.long().unique()
        else:  # Subsequent steps
            start_frame = self.model.num_execution_steps % self.model.key_frame_interval
            key_frames = torch.arange(
                start_frame, self.model.num_execution_steps+1, self.model.key_frame_interval, device=self.device
            )
            # exclude 0 from key_frames
            key_frames = key_frames[key_frames > 0]
            key_frames = key_frames.sort().values.long().unique()

        return key_frames

    def _apply_predictions(self, frame_offset):
        """Apply cached predictions to traffic vehicles (skip ego)"""
        if self.predictions is None:
            logger.warning("No predictions available to apply")
            return

        positions = self.predictions['positions']
        headings = self.predictions['headings']
        velocities = self.predictions['velocities']

        # Iterate through all traffic agents (skip ego at index 0)
        for scenario_id, agent_idx in self._scenario_id_to_agent_idx.items():
            if scenario_id == self.sdc_scenario_id:
                continue  # Skip ego vehicle

            if scenario_id not in self._scenario_id_to_obj_id:
                continue  # Vehicle not spawned yet

            # Extract prediction for this agent at this frame
            pos = positions[agent_idx, frame_offset].cpu().numpy()
            heading = headings[agent_idx, frame_offset].cpu().item()
            vel = velocities[agent_idx, frame_offset].cpu().numpy()

            # Apply state
            obj_id = self._scenario_id_to_obj_id[scenario_id]
            if obj_id not in self.spawned_objects:
                continue  # Object has been destroyed, skip
            obj = self.spawned_objects[obj_id]
            obj.set_position(pos)
            obj.set_heading_theta(heading)
            obj.set_velocity(vel)

    @property
    def ego_vehicle(self):
        return self.engine.agents[DEFAULT_AGENT]
