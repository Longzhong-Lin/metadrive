#%%
import os
import pickle
from typing import Union
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.idm_policy import IDMPolicy, TrajectoryIDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.component.navigation_module.traj_network_navigation import TrajNetworkNavigation
from metadrive.scenario.utils import get_number_of_scenarios

DATA_DIR = "/home/linlongzhong/Data/Datasets/Waymo/Motion/scenarionet/validation_filtered"
SCENARIO_IDX = 1000
EGO_POLICY: Union[ReplayEgoCarPolicy, IDMPolicy, TrajectoryIDMPolicy, ExpertPolicy] = ExpertPolicy

TRAFFIC_MODE = "unimm"  # "log", "idm", "unimm"
UNIMM_CKPT = "/home/linlongzhong/Data/Projects/QCNet_MA-Sim/output_archive/waymo/simulation_gpt/training/mlp-decoder_2048-anchors-kmeans-8s_pre-match_predict-4s_match-0.5s_match-scorer_sim-2Hz-8s_b32_e30/lightning_logs/version_0/checkpoints/last.ckpt"

EXP_MODE = "record" # "record", "replay"
EXP_DIR = "exp/unimm_traffic_demo"
REPLAY_FILE = "exp/unimm_traffic_demo/scenario-1000_24fa0f8655098ed8.pkl"  # Used when EXP_MODE is "replay"
REPLAY_3D = False

os.makedirs(EXP_DIR, exist_ok=True)
if EXP_MODE == "replay":
    with open(REPLAY_FILE, "rb") as f:
        replay_episode = pickle.load(f)

env = ScenarioEnv({
    "data_directory": DATA_DIR,
    "num_scenarios": get_number_of_scenarios(DATA_DIR),
    "vehicle_config": {"navigation_module": TrajNetworkNavigation},
    "agent_policy": EGO_POLICY,
    "reactive_traffic": True if TRAFFIC_MODE == "idm" else False,
    "use_unimm_traffic": True if TRAFFIC_MODE == "unimm" else False,
    "unimm_checkpoint": UNIMM_CKPT,
    "unimm_device": "cuda",
    "horizon": 90,
    "record_episode": True if EXP_MODE == "record" else False,
    "replay_episode": replay_episode if EXP_MODE == "replay" else {},
    "use_render": REPLAY_3D,
})

try:
    if EXP_MODE == "record":
        o, _ = env.reset(seed=SCENARIO_IDX)
        scenario_id = env.engine.data_manager.current_scenario_id
        
        for time_step in range(env.config["horizon"]):
            o, r, tm, tc, info = env.step([0.0, 0.0])
            env.render(
                mode="top_down",
                semantic_map=False,
                window=False,
                scaling=5.0,
                screen_size=(500, 500),
                draw_navi_info=True,
                text={"Time Step": time_step},
                screen_record=True,
            )
        
        record_path = os.path.join(EXP_DIR, f"scenario-{SCENARIO_IDX}_{scenario_id}.pkl")
        env.engine.dump_episode(record_path)
        print(f"✓ Record saved to: {record_path}")
        
        gif_path = os.path.join(EXP_DIR, f"scenario-{SCENARIO_IDX}_{scenario_id}.gif")
        env.top_down_renderer.generate_gif(gif_path)
        print(f"✓ GIF saved to: {gif_path}")
        
    elif EXP_MODE == "replay":
        o, _ = env.reset()
        scenario_id = env.engine.data_manager.current_scenario_id
        
        time_step = 0
        while True:
            o, r, tm, tc, info = env.step([0.0, 0.0])
            if REPLAY_3D:
                # 3D render
                env.render(
                    text={"Time Step": time_step},
                )
            else:
                # 2D render
                env.render(
                    mode="top_down",
                    semantic_map=False,
                    window=False,
                    scaling=5.0,
                    screen_size=(500, 500),
                    draw_navi_info=True,
                    text={"Time Step": time_step},
                    screen_record=True,
                )
            # Check if replay finished
            time_step += 1
            if info.get("replay_done", False):
                print(f"\n✓ Replay completed ({time_step} steps)")
                break

        if not REPLAY_3D:
            gif_path = os.path.join(EXP_DIR, f"scenario-{SCENARIO_IDX}_{scenario_id}_replay.gif")
            env.top_down_renderer.generate_gif(gif_path)
            print(f"✓ GIF saved to: {gif_path}")
    
    else:
        raise NotImplementedError
    
finally:
    env.close()
