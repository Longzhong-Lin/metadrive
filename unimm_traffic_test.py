"""
Test script for evaluating different traffic flows and ego policies in MetaDrive ScenarioEnv.

This script tests combinations of:
- Traffic modes: log replay, trajectory IDM, UniMM
- Ego policies: trajectory IDM, UniMM (expert)

Metrics evaluated:
- Collision: collision with vehicles, objects, humans
- Offroad: out of road violations
- Progress: route completion rate
- Comfort: acceleration/jerk statistics
"""

import argparse
import json
import os
import time
import torch
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.idm_policy import TrajectoryIDMPolicy
from metadrive.policy.expert_policy import ExpertPolicy
from metadrive.component.navigation_module.traj_network_navigation import TrajNetworkNavigation
from metadrive.scenario.utils import get_number_of_scenarios


class MetricsCollector:
    """Collects and computes metrics during episode evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset metrics for new episode."""

        # Collision
        self.collision = False
        self.crash_vehicle = False
        self.crash_object = False
        self.crash_building = False
        self.crash_human = False

        # Offroad
        self.offroad = False
        self.crash_sidewalk = False        
        self.out_of_road = False
        
        # Failure
        self.crash = False
        
        # Success
        self.arrive_dest = False
        
        # Progress
        self.route_completion = 0.0
        self.steps = 0

        # Traffic speeds
        self.traffic_speeds = []

    def update(self, info: Dict):
        """Update metrics at each step.

        Args:
            info: Info dict from env.step()
        """

        # Collision
        self.crash_vehicle = info.get('crash_vehicle', False) or self.crash_vehicle
        self.crash_object = info.get('crash_object', False) or self.crash_object
        self.crash_building = info.get('crash_building', False) or self.crash_building
        self.crash_human = info.get('crash_human', False) or self.crash_human
        self.collision = (
            self.collision or
            self.crash_vehicle or
            self.crash_object or
            self.crash_building or
            self.crash_human
        )

        # Offroad
        self.crash_sidewalk = info.get('crash_sidewalk', False) or self.crash_sidewalk
        self.out_of_road = info.get('out_of_road', False) or self.out_of_road
        self.offroad = (
            self.offroad or
            self.crash_sidewalk
        )
        
        # Failure
        self.crash = info.get('crash', False) or self.crash
        
        # Success
        self.arrive_dest = info.get('arrive_dest', False) or self.arrive_dest
        
        # Progress
        self.route_completion = info.get('route_completion', 0.0)
        self.steps += 1

        # Traffic speeds
        self.traffic_speeds.append(info.get('traffic_speed', 0.0))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected metrics.

        Returns:
            Dictionary containing all metrics
        """

        return {
            # Collision
            'collision': self.collision,
            'crash_vehicle': self.crash_vehicle,
            'crash_object': self.crash_object,
            'crash_building': self.crash_building,
            'crash_human': self.crash_human,
            
            # Offroad
            'offroad': self.offroad,
            'crash_sidewalk': self.crash_sidewalk,
            'out_of_road': self.out_of_road,

            # Failure
            'crash': self.crash,

            # Success
            'arrive_dest': self.arrive_dest,

            # Progress
            'route_completion': self.route_completion,
            'steps': self.steps,

            # Traffic speed
            'traffic_speed': np.mean(self.traffic_speeds),
        }


def create_env_config(
    traffic_mode: str,
    ego_policy: str,
    data_directory: str,
    unimm_checkpoint: str = None,
) -> Dict:
    """Create environment configuration for given traffic mode and ego policy.

    Args:
        traffic_mode: One of 'log', 'idm', 'unimm'
        ego_policy: One of 'idm', 'expert'
        data_directory: Path to scenario data
        unimm_checkpoint: Path to UniMM checkpoint (required for unimm mode)

    Returns:
        Configuration dictionary
    """
    # Base configuration
    config = {
        'data_directory': data_directory,
        'num_scenarios': get_number_of_scenarios(data_directory),
        "vehicle_config": {"navigation_module": TrajNetworkNavigation},
        'horizon': 90,  # max steps per episode
        'use_render': False,
    }

    # Traffic mode configuration
    if traffic_mode == 'log':
        # Log replay: use recorded trajectories
        config.update({
            'reactive_traffic': False,
            'use_unimm_traffic': False,
        })
    elif traffic_mode == 'idm':
        # Trajectory-based IDM traffic
        config.update({
            'reactive_traffic': True,
            'use_unimm_traffic': False,
        })
    elif traffic_mode == 'unimm':
        # UniMM-based traffic
        assert unimm_checkpoint is not None, "UniMM checkpoint is required for unimm traffic mode"
        config.update({
            'reactive_traffic': False,
            'use_unimm_traffic': True,
            'unimm_checkpoint': unimm_checkpoint,
            'unimm_device': 'cuda' if torch.cuda.is_available() else 'cpu',
        })
    else:
        raise ValueError(f"Unknown traffic mode: {traffic_mode}")

    # Ego policy configuration
    if ego_policy == 'idm':
        config['agent_policy'] = TrajectoryIDMPolicy
    elif ego_policy == 'expert':
        config['agent_policy'] = ExpertPolicy
    else:
        raise ValueError(f"Unknown ego policy: {ego_policy}")

    return config


def test_single_combination(
    traffic_mode: str,
    ego_policy: str,
    data_directory: str,
    scenario_indices: List[int],
    unimm_checkpoint: str = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Test a single combination of traffic mode and ego policy.

    Args:
        traffic_mode: Traffic mode to test
        ego_policy: Ego policy to test
        data_directory: Path to scenario data
        scenario_indices: List of scenario indices to test
        unimm_checkpoint: Path to UniMM checkpoint
        verbose: Whether to print detailed info

    Returns:
        List of metric dictionaries, one per scenario
    """
    # Create environment
    config = create_env_config(
        traffic_mode=traffic_mode,
        ego_policy=ego_policy,
        data_directory=data_directory,
        unimm_checkpoint=unimm_checkpoint,
    )

    env = ScenarioEnv(config)
    metrics = MetricsCollector()
    results = []

    try:
        for scenario_idx in tqdm(scenario_indices, desc=f"{traffic_mode} + {ego_policy}"):
            # Reset for new scenario
            env.reset(seed=int(scenario_idx))
            metrics.reset()

            for time_step in range(env.config["horizon"]):
                # Step environment
                obs, reward, terminated, truncated, info = env.step([0, 0])  # Policy handles action

                # Calculate traffic average speed
                traffic_objects = [
                    obj for obj_id, obj in env.engine.traffic_manager.spawned_objects.items()
                    if obj_id != env.agent.id
                ]
                if len(traffic_objects) > 0:
                    traffic_speed = np.mean([obj.speed for obj in traffic_objects])
                else:
                    traffic_speed = 0.0
                info['traffic_speed'] = traffic_speed

                # Update metrics
                metrics.update(info)

                # Check if episode is done
                if metrics.crash or metrics.arrive_dest or truncated:
                    break

            # Get episode summary
            episode_result = {
                'traffic_mode': traffic_mode,
                'ego_policy': ego_policy,
                'scenario_idx': scenario_idx,
            }
            episode_result.update(metrics.get_summary())
            results.append(episode_result)

            if verbose:
                print(
                    f"Scenario {scenario_idx} completed:"
                    f"collision={episode_result['collision']},"
                    f"offroad={episode_result['offroad']},"
                    f"success={episode_result['arrive_dest']},"
                    f"progress={episode_result['route_completion']:.2%},"
                )
    finally:
        env.close()

    return results


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate results from multiple scenarios.

    Args:
        results: List of per-scenario results

    Returns:
        Aggregated statistics
    """
    df = pd.DataFrame(results)

    # Aggregate metrics
    agg = {
        'num_scenarios': len(results),

        # Collision statistics
        'collision_rate': df['collision'].mean(),
        'crash_vehicle_rate': df['crash_vehicle'].mean(),
        'crash_object_rate': df['crash_object'].mean(),
        'crash_building_rate': df['crash_building'].mean(),
        'crash_human_rate': df['crash_human'].mean(),

        # Offroad statistics
        'offroad_rate': df['offroad'].mean(),
        'crash_sidewalk_rate': df['crash_sidewalk'].mean(),
        'out_of_road_rate': df['out_of_road'].mean(),

        # Failure statistics
        'crash_rate': df['crash'].mean(),

        # Success statistics
        'success_rate': df['arrive_dest'].mean(),

        # Progress statistics
        'route_completion_mean': df['route_completion'].mean(),
        'route_completion_std': df['route_completion'].std(),
        'num_steps': df['steps'].mean(),

        # Traffic speed statistics
        'traffic_speed_mean': df['traffic_speed'].mean(),
        'traffic_speed_std': df['traffic_speed'].std(),
    }

    return agg


def save_results(
    all_results: Dict[str, List[Dict]],
    aggregated: Dict[str, Dict],
    output_dir: str,
    timestamp: str,
):
    """Save results to files.

    Args:
        all_results: Dictionary mapping (traffic_mode, ego_policy) to results list
        aggregated: Dictionary mapping (traffic_mode, ego_policy) to aggregated stats
        output_dir: Output directory
        timestamp: Timestamp string for filenames
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed results as CSV
    all_detailed = []
    for (traffic_mode, ego_policy), results in all_results.items():
        all_detailed.extend(results)

    if all_detailed:
        df = pd.DataFrame(all_detailed)
        csv_path = os.path.join(output_dir, f'detailed_results_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed results to {csv_path}")

    # Save aggregated results as JSON
    json_data = {}
    for (traffic_mode, ego_policy), stats in aggregated.items():
        key = f"{traffic_mode}_{ego_policy}"
        json_data[key] = stats

    json_path = os.path.join(output_dir, f'aggregated_results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved aggregated results to {json_path}")

    # Save summary table as CSV
    summary_rows = []
    for (traffic_mode, ego_policy), stats in aggregated.items():
        row = {
            'traffic_mode': traffic_mode,
            'ego_policy': ego_policy,
            'collision_rate': f"{stats.get('collision_rate', 0):.3f}",
            'offroad_rate': f"{stats.get('offroad_rate', 0):.3f}",
            'success_rate': f"{stats.get('success_rate', 0):.3f}",
            'avg_progress': f"{stats.get('route_completion_mean', 0):.3f}",
            'traffic_speed': f"{stats.get('traffic_speed_mean', 0):.2f}",
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, f'summary_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table to {summary_path}")

    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Test different traffic flows and ego policies')

    # Data configuration
    parser.add_argument('--data_directory', type=str, required=True, help='Path to scenario data directory')
    parser.add_argument('--num_scenarios', type=int, default=5, help='Number of scenarios to randomly sample and test (default: 5)')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for scenario sampling (default: 0)')

    # Test configuration
    parser.add_argument(
        '--traffic_modes', type=str, nargs='+', default=['log', 'idm', 'unimm'],
        choices=['log', 'idm', 'unimm'], help='Traffic modes to test (default: all)'
    )
    parser.add_argument(
        '--ego_policies', type=str, nargs='+', default=['idm', 'expert'],
        choices=['idm', 'expert'], help='Ego policies to test (default: all)'
    )
    parser.add_argument(
        '--unimm_checkpoint', type=str, default=None,
        help='Path to UniMM checkpoint (required for unimm traffic)'
    )

    # Output configuration
    parser.add_argument(
        '--output_dir', type=str, default='./exp/unimm_traffic_test',
        help='Output directory for results (default: ./exp/unimm_traffic_test)'
    )

    # Other options
    parser.add_argument('--verbose', action='store_true', help='Print detailed progress')

    args = parser.parse_args()

    # Validate UniMM checkpoint
    if 'unimm' in args.traffic_modes and args.unimm_checkpoint is None:
        parser.error("--unimm_checkpoint is required when testing 'unimm' traffic mode")

    print("="*80)
    print("MetaDrive Traffic & Policy Testing")
    print("="*80)
    print(f"Data directory: {args.data_directory}")
    print(f"Number of scenarios to sample: {args.num_scenarios}")
    print(f"Random seed: {args.random_seed}")
    print(f"Traffic modes: {args.traffic_modes}")
    print(f"Ego policies: {args.ego_policies}")
    print(f"Output directory: {args.output_dir}")
    print("-"*80 + "\n")

    # Generate test combinations
    combinations = [
        (traffic_mode, ego_policy)
        for traffic_mode in args.traffic_modes
        for ego_policy in args.ego_policies
    ]

    print(f"Total combinations to test: {len(combinations)}\n")

    # Sample scenarios once for all combinations
    total_scenarios = get_number_of_scenarios(args.data_directory)
    np.random.seed(args.random_seed)
    sampled_indices = np.random.choice(
        total_scenarios,
        size=min(args.num_scenarios, total_scenarios),
        replace=False
    ).tolist()

    print(f"Total scenarios available: {total_scenarios}")
    print(f"Sampled scenario indices: {sampled_indices}\n")

    # Run tests
    all_results = {}
    aggregated = {}

    start_time = time.time()

    for traffic_mode, ego_policy in combinations:
        print("-" * 80)
        print(f"Testing: {traffic_mode} + {ego_policy}\n")

        results = test_single_combination(
            traffic_mode=traffic_mode,
            ego_policy=ego_policy,
            data_directory=args.data_directory,
            scenario_indices=sampled_indices,
            unimm_checkpoint=args.unimm_checkpoint,
            verbose=args.verbose,
        )

        all_results[(traffic_mode, ego_policy)] = results

        # Aggregate results
        agg = aggregate_results(results)
        aggregated[(traffic_mode, ego_policy)] = agg

        print(f"Completed: {traffic_mode} + {ego_policy}")
        print(f"  Collision rate: {agg.get('collision_rate', 0):.2%}")
        print(f"  Offroad rate: {agg.get('offroad_rate', 0):.2%}")
        print(f"  Success rate: {agg.get('success_rate', 0):.2%}")
        print(f"  Avg progress: {agg.get('route_completion_mean', 0):.2%}")
        print(f"  Traffic speed: {agg.get('traffic_speed_mean', 0):.2f} m/s")

    elapsed_time = time.time() - start_time
    print(f"\nTotal testing time: {elapsed_time:.2f} seconds")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_results(all_results, aggregated, args.output_dir, timestamp)

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
