#%%
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.policy.idm_policy import IDMPolicy, TrajectoryIDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.component.navigation_module.traj_network_navigation import TrajNetworkNavigation
from IPython.display import Image

env = ScenarioEnv({
    "data_directory": "/home/linlongzhong/Data/Datasets/Waymo/Motion/scenarionet/validation_filtered",
    "num_scenarios": 6758,
    "vehicle_config": {"navigation_module": TrajNetworkNavigation},
    "agent_policy": ExpertPolicy,
    "reactive_traffic": False,
    "use_unimm_traffic": True,
    "unimm_checkpoint": "/home/linlongzhong/Data/Projects/QCNet_MA-Sim/output_archive/waymo/simulation_gpt/training/mlp-decoder_2048-anchors-kmeans-8s_pre-match_predict-4s_match-0.5s_match-scorer_sim-2Hz-8s_b32_e30/lightning_logs/version_0/checkpoints/last.ckpt",
    "unimm_device": "cuda",
    "horizon": 90,
})

try:
    o, _ = env.reset(seed=1000)
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
    env.top_down_renderer.generate_gif()
finally:
    env.close()

Image("demo.gif")