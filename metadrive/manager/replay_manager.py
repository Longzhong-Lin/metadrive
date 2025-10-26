import copy
import logging

from metadrive.base_class.base_object import BaseObject
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import PGMap, MapGenerateMethod
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.constants import DEFAULT_AGENT
from metadrive.constants import ObjectState
from metadrive.constants import PolicyState
from metadrive.constants import REPLAY_DONE
from metadrive.manager.base_manager import BaseManager
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.utils import recursive_equal


class ReplayManager(BaseManager):
    PRIORITY = 99  # lowest

    def __init__(self):
        super(ReplayManager, self).__init__()
        self.restore_episode_info = None
        self.current_map = None
        self.current_frames = None
        self.current_frame = None
        self.replay_done = False
        self.record_name_to_current_name = dict()
        self.current_name_to_record_name = dict()
        self.observation = self.get_observation()

    def before_reset(self, *args, **kwargs):
        """
        Clean generated objects
        """
        if self.engine.only_reset_when_replay:
            for name in self.spawned_objects.keys():
                assert name not in self.engine._spawned_objects, \
                    "Other Managers failed to clean objects loaded by ReplayManager"
            self.spawned_objects = {}
            assert len(self.engine._object_policies) == 0, "Policy should be cleaned for reducing memory usage"
        else:
            self.clear_objects([name for name in self.spawned_objects])
        self.replay_done = False

    def spawn_object(self, object_class, **kwargs):
        """
        Spawn one objects
        """
        object = self.engine.spawn_object(object_class, **kwargs)
        if not isinstance(object, BaseMap):
            # map has different treatment as what has been done in MapManager
            self.spawned_objects[object.id] = object
        return object

    def reset(self):
        if not self.engine.replay_episode:
            return
        assert not self.engine.record_episode, "When replay, please set record to False"
        self.record_name_to_current_name = dict()
        self.current_name_to_record_name = dict()
        self.restore_episode_info = self.engine.global_config["replay_episode"]
        self.restore_episode_info["frame"].reverse()
        # Since in episode data map data only contains one map, values()[0] is the map_parameters
        map_data = self.restore_episode_info["map_data"]
        assert len(map_data) > 0, "Can not find map info in episode data"

        if "map_config" in map_data and "block_sequence" in map_data:
            # This is a PGMap (procedurally generated map)
            map_config = copy.deepcopy(map_data["map_config"])
            map_config[BaseMap.GENERATE_TYPE] = MapGenerateMethod.PG_MAP_FILE
            map_config[BaseMap.GENERATE_CONFIG] = map_data["block_sequence"]
            if self.engine.map_manager.maps[self.engine.global_seed] is not None:
                self.current_map = self.engine.map_manager.maps[self.engine.global_seed]
                assert recursive_equal(
                    self.current_map.get_meta_data()["block_sequence"], map_data["block_sequence"], need_assert=True
                ), "Loaded data mismatch stored data"
                self.engine.map_manager.load_map(self.current_map)
            else:
                self.current_map = self.spawn_object(
                    PGMap, map_config=map_config, auto_fill_random_seed=False, force_spawn=True
                )
        else:
            # This is a ScenarioMap - load it from the scenario data
            # For ScenarioEnv replay, we need to ensure the map is loaded from the scenario data
            if hasattr(self.engine, 'data_manager') and hasattr(self.engine, 'map_manager'):
                # Import here to avoid circular dependency
                from metadrive.manager.scenario_map_manager import ScenarioMapManager
                from metadrive.component.map.scenario_map import ScenarioMap

                # For ScenarioEnv, the map manager needs to be reset to load the correct scenario map
                if isinstance(self.engine.map_manager, ScenarioMapManager):
                    # Get the scenario ID from the replay episode info
                    scenario_index = self.restore_episode_info.get("scenario_index", None)
                    if scenario_index is None:
                        scenario_index = self.restore_episode_info.get("global_seed", None)

                    # Ensure the engine's global seed is set correctly for the scenario
                    if scenario_index is not None:
                        self.engine.global_random_seed = scenario_index

                    # Reset the map manager to load the scenario map
                    self.engine.map_manager.reset()
                    self.current_map = self.engine.map_manager.current_map

                    if self.current_map is None or not isinstance(self.current_map, ScenarioMap):
                        raise ValueError(
                            f"Failed to load ScenarioMap for scenario index {scenario_index}. "
                            "The map_manager did not create a ScenarioMap."
                        )
                else:
                    raise ValueError(
                        "ScenarioMap replay requires ScenarioMapManager. "
                        f"Found {type(self.engine.map_manager).__name__} instead."
                    )
            else:
                raise ValueError(
                    "ScenarioMap replay requires both data_manager and map_manager to be available. "
                    "Please ensure the environment is properly initialized."
                )

        self.current_frames = self.restore_episode_info["frame"].pop()
        self.replay_frame()
        if self.engine.only_reset_when_replay:
            # do not replay full trajectory! set state for managers for interaction
            self.restore_manager_states(self.current_frame.manager_info)
            # Do special treatment to map manager
            self.engine.map_manager.current_map = self.current_map
            if hasattr(self.engine.map_manager, 'maps'):
                # Only PGMapManager has the maps dict
                self.engine.map_manager.maps[self.engine.global_seed] = self.current_map

    def restore_policy_states(self, policy_spawn_infos):
        # restore agent policy
        agent_policy = self.engine.agent_manager.agent_policy
        agent_obj_name = self.engine.agent_manager.active_agents[DEFAULT_AGENT].name
        for name, policy_spawn_info in policy_spawn_infos.items():
            obj_name = self.record_name_to_current_name[name]
            p_class = policy_spawn_info[PolicyState.POLICY_CLASS] if obj_name != agent_obj_name else agent_policy
            args = policy_spawn_info[PolicyState.ARGS]
            kwargs = policy_spawn_info[PolicyState.KWARGS]
            assert obj_name == self.record_name_to_current_name[policy_spawn_info[PolicyState.OBJ_NAME]]
            assert obj_name in self.engine.get_objects().keys(), "Can not find obj when restoring policies"
            policy = self.add_policy(obj_name, p_class, *args, **kwargs)
            if policy.control_object is BaseObject:
                obj = list(self.engine.get_objects([obj_name]).values())[0]
                policy.control_object = obj
                assert obj.id == obj_name

    def restore_manager_states(self, states):
        current_managers = [manager.class_name for manager in self.engine.managers.values()]
        data_managers = states.keys()
        assert len(current_managers - data_managers
                   ) == 0, "Manager not match, data: {}, current: {}".format(data_managers, current_managers)
        for manager in self.engine.managers.values():
            manager.set_state(states[manager.class_name], old_name_to_current=self.record_name_to_current_name)

    def step(self, *args, **kwargs):
        # Note: Update object state must be written in step, because the simulator will step 5 times for each RL step.
        # We need to record the intermediate states.
        if self.engine.replay_episode and not self.engine.only_reset_when_replay:
            self.replay_frame()

    def after_step(self, *args, **kwargs):
        if self.engine.replay_episode and not self.engine.only_reset_when_replay:
            if len(self.restore_episode_info["frame"]) == 0:
                self.replay_done = True
            return self.engine.agent_manager.for_each_active_agents(lambda v: {REPLAY_DONE: self.replay_done})
        else:
            return dict()

    def destroy(self):
        self.record_name_to_current_name = dict()
        self.current_name_to_record_name = dict()
        self.restore_episode_info = None
        self.current_map = None

    def replay_frame(self):
        if len(self.current_frames) == 0:
            return
        self.current_frame = self.current_frames.pop()
        # create
        for name, config in self.current_frame.spawn_info.items():
            if config[ObjectState.CLASS] == DefaultVehicle:
                config[ObjectState.INIT_KWARGS]["vehicle_config"]["use_special_color"] = True
            obj = self.spawn_object(object_class=config[ObjectState.CLASS], **config[ObjectState.INIT_KWARGS])
            self.current_name_to_record_name[obj.name] = name
            self.record_name_to_current_name[name] = obj.name
            if issubclass(config[ObjectState.CLASS], BaseVehicle):
                # For PGMap, set route using spawn_road and destination
                # For ScenarioMap, these fields may not exist
                step_info = self.current_frame.step_info.get(name, {})
                if "spawn_road" in step_info and "destination" in step_info:
                    obj.navigation.set_route(
                        step_info["spawn_road"],
                        step_info["destination"][-1]
                    )
        if self.engine.only_reset_when_replay:
            # for generation policies
            self.restore_policy_states(self.current_frame.policy_spawn_info)
        else:
            # Do not set position, in this mode, or randomness will be introduced!
            for name, state in self.current_frame.step_info.items():
                self.spawned_objects[self.record_name_to_current_name[name]].before_step()
                self.spawned_objects[self.record_name_to_current_name[name]].set_state(state)
                self.spawned_objects[self.record_name_to_current_name[name]].after_step()

        to_clear = []
        for name in self.current_frame.clear_info:
            to_clear.append(self.record_name_to_current_name[name])
            self.current_name_to_record_name.pop(self.record_name_to_current_name[name])
            self.record_name_to_current_name.pop(name)
        self.clear_objects(to_clear)

        self.replay_done = False

    @property
    def replay_agents(self):
        return {k: self.get_object_from_agent(k) for k in self.current_frame.agents}

    def __del__(self):
        logging.debug("Replay system is destroyed")

    def get_object_from_agent(self, agent_id):
        return self.spawned_objects[self.record_name_to_current_name[self.current_frame.agent_to_object(agent_id)]]

    def get_observation(self):
        """
        Override me in the future for collecting other modality
        """
        return LidarStateObservation(self.engine.global_config)

    def get_replay_agent_observations(self):
        return {k: self.observation for k in self.current_frame.agents}

    def set_state(self, state: dict, old_name_to_current=None):
        return {}

    def get_state(self):
        return {}

    def before_step(self, *args, **kwargs) -> dict:
        super(ReplayManager, self).before_step()
        if self.engine.replay_episode and not self.engine.only_reset_when_replay:
            self.current_frames = self.restore_episode_info["frame"].pop()
            self.current_frames.reverse()
        return {}
