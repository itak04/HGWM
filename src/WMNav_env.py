import gzip
import json
import logging
import math
import os
import random
import requests
import time
import traceback
import habitat_sim

import pandas as pd
import numpy as np

from PIL import Image
from simWrapper import PolarAction, SimWrapper
from WMNav_agent import *
from custom_agent import *
from utils import *

class Env:
    """
    Base class for creating an environment for embodied navigation tasks.
    This class defines the setup, logging, running, and evaluation of episodes.
    """

    task = 'Not defined'

    def __init__(self, cfg: dict):
        """
        Initializes the environment with the provided configuration.

        Args:
            cfg (dict): Configuration dictionary containing environment, simulation, and agent settings.
        """
        self.cfg = cfg['env_cfg']
        self.sim_cfg = cfg['sim_cfg']
        if self.cfg['name'] == 'default':
            self.cfg['name'] = f'default_{random.randint(0, 1000)}'
        self._initialize_logging(cfg)
        self._initialize_agent(cfg)
        self.outer_run_name = self.task + '_' + self.cfg['name']
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper = None  # 修改self.simWrapper: SimWrapper = None
        self.num_episodes = 0
        self._initialize_experiment()

    def _initialize_agent(self, cfg: dict):
        """Initializes the agent for the environment."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        agent_cls = globals()[cfg['agent_cls']]
        self.agent: Agent = agent_cls(cfg['agent_cfg'])
        self.agent_cls = cfg['agent_cls']

    def _initialize_logging(self, cfg: dict):
        """
        确保日志目录始终存在的版本
        """
        # 首先确保LOG_DIR环境变量存在
        log_dir = os.environ.get("LOG_DIR")
        if not log_dir:
            log_dir = "/home/ps/dqf/GoalNav/WMNavigation/logs"
            os.environ["LOG_DIR"] = log_dir
            print(f"⚠️  LOG_DIR not set, using default: {log_dir}")
        
        # 确保LOG_DIR目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            print(f"📁 Created LOG_DIR: {log_dir}")
        else:
            print(f"📁 LOG_DIR exists: {log_dir}")
        
        # 设置日志文件路径
        task_name = cfg.get("task", "Unknown")
        env_name = self.cfg.get("name", "default")
        instance = self.cfg.get("instance", 0)
        instances = self.cfg.get("instances", 1)
        
        log_subdir = f'{task_name}_{env_name}'
        log_filename = f'{instance}_of_{instances}.txt'
        
        # 创建完整的日志目录结构
        full_log_dir = os.path.join(log_dir, log_subdir)
        os.makedirs(full_log_dir, exist_ok=True)
        
        self.log_file = os.path.join(full_log_dir, log_filename)
        
        print(f"📋 Log file will be: {self.log_file}")
        print(f"📁 Log directory structure: {full_log_dir}")
        
        if self.cfg['parallel']:
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )

    def _initialize_experiment(self):
        """
        Abstract method for setting up the environment and initializing all required variables.
        Should be implemented in derived classes.
        """
        raise NotImplementedError

    def run_experiment(self):
        """
        Runs the experiment by iterating over episodes.
        """
        instance_size = math.ceil(self.num_episodes / self.cfg['instances'])
        start_ndx = self.cfg['instance'] * instance_size
        end_ndx = self.num_episodes

        for episode_ndx in range(start_ndx, min(start_ndx + self.cfg['num_episodes'], end_ndx)):
            self.wandb_log_data = {
                'episode_ndx': episode_ndx,
                'instance': self.cfg['instance'],
                'total_episodes': self.cfg['instances'] * self.cfg['num_episodes'],
                'task': self.task,
                'task_data': {},
                'spl': 0,
                'goal_reached': False
            }

            try:
                self._run_episode(episode_ndx)
            except Exception as e:
                log_exception(e)
                self.simWrapper.reset()


    def _run_episode(self, episode_ndx: int):
        """
        Runs a single episode.p

        Args:
            episode_ndx (int): The index of the episode to run.
        """
        obs = self._initialize_episode(episode_ndx)  # color_sensor(1080, 1920, 4) depth_sensor(1080, 1920) agent_state[position rotation sensor_states]

        logging.info(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(self.cfg['max_steps']):
            try:
                agent_action = self._step_env(obs)  # 根据单张RGB图片、深度图和agent以及相机位姿确定agent的下一步动作，保存运行结果
                if agent_action is None:
                    break
                obs = self.simWrapper.step(agent_action)  # 执行操作，更新agent的状态和观察

            except Exception as e:
                log_exception(e)

            finally:
                self.step += 1
        self._post_episode()

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode. This method should be implemented in derived classes.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        self.step = 0
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0
        self.prev_agent_position = None

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment. This method should be implemented in derived classes.

        Args:
            obs (dict): The current observation. Contains agent state and sensor observations.

        Returns:
            PolarAction: The next action to be taken by the agent.
        """
        logging.info(f'Step {self.step}')
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position

        return None

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        # Ensure log directory exists before saving
        log_dir = os.environ.get("LOG_DIR")
        episode_log_dir = os.path.join(log_dir, f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}')
        os.makedirs(episode_log_dir, exist_ok=True)
        
        # Save logs and reset environment
        self.df.to_pickle(os.path.join(episode_log_dir, 'df_results.pkl'))
        self.simWrapper.reset()
        self.agent.reset()
        
        # Send data to aggregator if in parallel mode
        if self.cfg['parallel']:
            try:
                # Prepare metrics
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                
                # Add timeout and retry logic
                for attempt in range(3):  # Try up to 3 times
                    try:
                        response = requests.post(
                            f'http://localhost:{self.cfg["port"]}/log', 
                            json=self.wandb_log_data,
                            timeout=5  # Add 5 second timeout
                        )
                        if response.status_code == 200:
                            break  # Success, exit retry loop
                        logging.warning(f"Attempt {attempt+1}: Failed to send metrics: {response.status_code}")
                    except requests.exceptions.RequestException as req_err:
                        logging.warning(f"Attempt {attempt+1}: Connection error: {req_err}")
                        if attempt < 2:  # Don't sleep on last attempt
                            time.sleep(1)  # Wait before retrying
                
            except Exception as e:
                logging.error(f"Error in post-episode processing: {e}")
                for frame in traceback.extract_tb(e.__traceback__):
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
    
        # Continue with normal processing regardless of aggregator status
        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')
        
        # Generate GIFs if needed
        if self.cfg['log_freq'] == 1:
            create_gif(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'), 
                self.agent.cfg['sensor_cfg']['img_height'], 
                self.agent.cfg['sensor_cfg']['img_width'], 
                agent_cls=self.agent_cls
            )
            create_gif_nav(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )
            create_gif_cvalue(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        """
        Appends the step metadata to the dataframe, and saves the images and general metadata to disk.

        Args:
            images (dict): Images generated during the step.
            step_metadata (dict): Metadata for the current step.
            logging_data (dict): General logging data.
        """
        self.df = pd.concat([self.df, pd.DataFrame([step_metadata])], ignore_index=True)

        if self.step % self.cfg['log_freq'] == 0 or step_metadata['success'] == 0:
            path = os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}')
            if not step_metadata['success']:
                path += '_ERROR'
            os.makedirs(path, exist_ok=True)
            for name, im in images.items():
                if im is not None:
                    im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                    im.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w') as file:
                if step_metadata['success']:
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')

    def _calculate_metrics(self, agent_state: habitat_sim.AgentState, agent_action: PolarAction, geodesic_path: int, max_steps: int):
        """
        Calculates the navigation metrics at a given step.

        Args:
            agent_state: The state of the agent.
            agent_action: The action taken by the agent.
            geodesic_path: The shortest path to the goal.
            max_steps (int): Maximum steps allowed for the episode.

        Returns:
            dict: A dictionary containing calculated metrics.
        """
        metrics = {}
        self.path_calculator.requested_start = agent_state.position
        metrics['distance_to_goal'] = self.simWrapper.get_path(self.path_calculator)
        metrics['spl'] = 0
        metrics['goal_reached'] = False
        metrics['done'] = False
        metrics['finish_status'] = 'running'

        if agent_action is PolarAction.stop or self.step + 1 == max_steps:
            metrics['done'] = True

            if metrics['distance_to_goal'] < self.cfg['success_threshold']:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': metrics['goal_reached']
                })
            else:
                if agent_action is PolarAction.stop:
                    metrics['finish_status'] = 'fp'
                else:
                    metrics['finish_status'] = 'max_steps'

        return metrics

class WMNavEnv(Env):

    task = 'ObjectNav'

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.all_episodes = []
        if self.cfg['dataset']  == 'hm3d_v0.1':
            scene_config_path = 'hm3d_v0.1/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v1'
        elif self.cfg['dataset']  == 'hm3d_v0.2':
            scene_config_path = 'hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_hm3d_v2'
        elif self.cfg['dataset']  == 'mp3d':
            scene_config_path = 'mp3d/mp3d_annotated_basis.scene_dataset_config.json'
            objnav_path = 'objectnav_mp3d'
        else:
            raise ValueError('Dataset type must be hm3d_v0.1, hm3d_v0.2, or mp3d')

        self.sim_cfg['scene_config'] = os.path.join(os.environ.get("DATASET_ROOT"), scene_config_path)
        self.goals = {}
        
        dataset_path = os.path.join(os.environ.get("DATASET_ROOT"), objnav_path, f'{self.cfg["split"]}/content')
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        dataset_files = sorted(os.listdir(dataset_path))
        print(f"Loading {len(dataset_files)} dataset files...")

        for f in dataset_files:
            with gzip.open(os.path.join(dataset_path, f), 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals_by_category']
                self.all_episodes += js['episodes']

        self.num_episodes = len(self.all_episodes)
        print(f"Loaded {self.num_episodes} episodes from dataset")

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the BASE task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]
        if 'hm3d' in self.cfg['dataset']:
            f = episode['scene_id'].split('/')[1:]
            self.sim_cfg['scene_id'] = f[1][2:5]
            self.sim_cfg['scene_path'] = os.path.join(os.environ.get("DATASET_ROOT"), 'hm3d_v0.1' if self.cfg['dataset'] == 'hm3d_v0.1' else 'hm3d_v0.2', f'{self.cfg["split"]}/{f[1]}/{f[2]}')
            self.simWrapper = SimWrapper(self.sim_cfg)

            goals = self.goals[f[1][6:]]
            all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        elif 'mp3d' in self.cfg['dataset']:
            self.sim_cfg['scene_id'] = episode['scene_id'].split('/')[1]
            self.sim_cfg['scene_path'] = os.path.join(os.environ.get("DATASET_ROOT"), f'{episode["scene_id"]}')
            self.simWrapper = SimWrapper(self.sim_cfg)

            goals = self.goals[self.sim_cfg['scene_id']]
            all_objects = goals[f'{episode["scene_id"].split("/")[2]}_{episode["object_category"]}']
        else:
            raise ValueError('Dataset type must be hm3d_v0.1, hm3d_v0.2, or mp3d')
        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)
        logging.info(f'RUNNING EPISODE {episode_ndx} with {episode["object_category"]} and {len(all_objects)} instances. GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]}')
        if episode['object_category'] == 'tv_monitor':
            episode['object_category'] = 'tv screen'
        self.current_episode = {
            'object': episode['object_category'],
            'shortest_path': episode['info']['geodesic_distance'],
            'object_positions': [a['position'] for a in all_objects],
            'view_positions': view_positions
        }
        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

        # Create log directory for this episode early 
        log_dir = os.environ.get("LOG_DIR")
        episode_log_dir = os.path.join(log_dir, f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}')
        os.makedirs(episode_log_dir, exist_ok=True)
        print(f"📁 Created episode log directory: {episode_log_dir}")

        obs = self.simWrapper.step(PolarAction.null)

        self.previous_subtask = '{}'  # Initialize the last subtask with an empty dictionary
        return obs

    def _step_env(self, obs: dict):
        """
        Optimized step environment for CoTGraphAgent.
        Simplified panoramic navigation with CoT-based directional analysis.
        """
        episode_images = [(obs['color_sensor'].copy())[:, :, :3]]
        color_origin = episode_images[0]
        
        # Rotation actions for panoramic view
        loop_actions = {
            'clockwise': PolarAction(0, -0.167 * np.pi),
            'counterclock': PolarAction(0, 0.167 * np.pi)
        }
        
        # Collect panoramic images (12 views: 0°, 30°, 60°, ..., 330°)
        for i in range(11):
            obs = self.simWrapper.step(loop_actions['clockwise'])
            # For CoTGraphAgent, only update navigability for key directions
            if i % 2 == 0:
                self.agent.navigability(obs, i+1)
            episode_images.append((obs['color_sensor'].copy())[:, :, :3])
        
        # Generate navigation map
        nav_map = self.agent.generate_voxel(obs['agent_state'])
        
        # CoT-based panoramic analysis
        print(f"\n🎯 Starting CoT Panoramic Analysis for goal: {self.current_episode['object']}")
        panoramic_data = self.agent.make_curiosity_value(
            episode_images[-12:],  # Use all 12 panoramic images
            self.current_episode['object']
        )
        panoramic_image, explorable_value, reason = panoramic_data
        
        # Determine best direction using CoT analysis
        goal_rotate, goal_reason = self.agent.update_curiosity_value(explorable_value, reason)
        
        print(f"🔄 CoT Analysis - Optimal Direction: {goal_rotate * 30}°")
        
        # CoT-based planning
        pano_images = episode_images[-12:]
        try:
            # For CoTGraphAgent, use the optimal direction image for planning
            if isinstance(pano_images, list) and len(pano_images) > goal_rotate:
                target_image = [pano_images[goal_rotate]]
            else:
                target_image = pano_images
        except Exception as e:
            logging.warning(f"Error selecting target image: {e}")
            target_image = pano_images
        
        # Enhanced planning with CoT reasoning
        goal_flag, subtask = self.agent.make_plan(
            target_image, 
            self.previous_subtask, 
            goal_reason, 
            self.current_episode['object']
        )
        
        self.previous_subtask = subtask
        
        # Rotate to target direction
        for j in range(min(11 - goal_rotate, 1 + goal_rotate)):
            if goal_rotate <= 6:
                obs = self.simWrapper.step(loop_actions['clockwise'])
            else:
                obs = self.simWrapper.step(loop_actions['counterclock'])

        # Generate curiosity value map
        cvalue_map = self.agent.draw_cvalue_map(obs['agent_state'])
        
        # Update observation with CoT context
        super()._step_env(obs)
        obs.update({
            'goal': self.current_episode['object'],
            'subtask': subtask,
            'goal_flag': goal_flag
        })
        
        # Update agent position tracking
        agent_state = obs['agent_state']
        self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position
        
        # Enhanced logging for CoT analysis
        logging.info(f"CoT Analysis - Goal: {self.current_episode['object']}, Direction: {goal_rotate * 30}°, Flag: {goal_flag}")
        
        # Get agent action with CoT context
        agent_action, metadata = self.agent.step(obs)
        step_metadata = metadata['step_metadata']
        
        # Enhanced logging data for CoT
        cot_stats = {}
        if hasattr(self.agent, 'get_optimization_stats'):
            cot_stats = self.agent.get_optimization_stats()
        
        log_responses = {
            'COT_EVALUATOR_RESPONSE': {
                'goal_rotate_degrees': goal_rotate * 30,
                'explorable_value': explorable_value,
                'reason': reason,
                'cot_stats': cot_stats
            },
            'COT_PLANNING_RESPONSE': {
                'goal_flag': goal_flag,
                'subtask': subtask,
                'goal_reason': goal_reason
            }
        }
        
        for key, value in log_responses.items():
            metadata['logging_data'][key] = str(value)
        
        # Image processing and visualization
        images = metadata['images']
        
        # Add step and goal information
        if metadata.get('step') is not None:
            color_origin = self._add_text_to_image(color_origin, f"step {metadata['step']}", (10, 30))
        
        if obs.get('goal') is not None:
            color_origin = self._add_goal_text(color_origin, obs['goal'])
        
        # Combine visualization images
        planner_images = {
            'panoramic': panoramic_image,
            'color_origin': color_origin,
            'nav_map': nav_map,
            'cvalue_map': cvalue_map,
        }
        
        # Add CoT-specific visualizations
        if hasattr(self.agent, 'direction_analysis') and self.agent.direction_analysis:
            planner_images['cot_analysis'] = f"CoT Analysis: {len(self.agent.direction_analysis)} directions analyzed"
        
        images.update(planner_images)
        
        # Calculate metrics
        metrics = self._calculate_metrics(agent_state, agent_action, 
                                         self.current_episode['shortest_path'], self.cfg['max_steps'])
        step_metadata.update(metrics)
        
        # Log results
        self._log(images, step_metadata, metadata['logging_data'])
        
        # Check episode completion
        if metrics['done']:
            logging.info(f"Episode {self.curr_run_name} completed with status: {metrics['finish_status']}")
            if metrics['goal_reached']:
                logging.info(f"🎉 CoT Goal {obs['goal']} reached successfully!")
            else:
                logging.info(f"❌ CoT Goal {obs['goal']} not reached. Distance: {metrics['distance_to_goal']:.2f}m")
            agent_action = None
        
        return agent_action
    
    def _add_text_to_image(self, image, text, position, color=(255, 255, 255), font_scale=1, thickness=2):
        """优化：添加文本到图像的辅助方法"""
        image = np.ascontiguousarray(image)
        return cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    def _add_goal_text(self, image, goal):
        """优化：添加目标文本的辅助方法"""
        scale_factor = image.shape[0] / 1080
        padding = 20
        text_size = 2.5 * scale_factor
        text_thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(f"goal:{goal}", 
                                                     cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
        text_position = (image.shape[1] - text_width - padding, padding + text_height)
        
        return self._add_text_to_image(image, f"goal:{goal}", text_position, (255, 0, 0), text_size, text_thickness)

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        # Ensure log directory exists before saving
        log_dir = os.environ.get("LOG_DIR")
        episode_log_dir = os.path.join(log_dir, f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}')
        os.makedirs(episode_log_dir, exist_ok=True)
        
        # Save logs and reset environment
        self.df.to_pickle(os.path.join(episode_log_dir, 'df_results.pkl'))
        self.simWrapper.reset()
        self.agent.reset()
        
        # Send data to aggregator if in parallel mode
        if self.cfg['parallel']:
            try:
                # Prepare metrics
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)
                
                # Add timeout and retry logic
                for attempt in range(3):  # Try up to 3 times
                    try:
                        response = requests.post(
                            f'http://localhost:{self.cfg["port"]}/log', 
                            json=self.wandb_log_data,
                            timeout=5  # Add 5 second timeout
                        )
                        if response.status_code == 200:
                            break  # Success, exit retry loop
                        logging.warning(f"Attempt {attempt+1}: Failed to send metrics: {response.status_code}")
                    except requests.exceptions.RequestException as req_err:
                        logging.warning(f"Attempt {attempt+1}: Connection error: {req_err}")
                        if attempt < 2:  # Don't sleep on last attempt
                            time.sleep(1)  # Wait before retrying
                
            except Exception as e:
                logging.error(f"Error in post-episode processing: {e}")
                for frame in traceback.extract_tb(e.__traceback__):
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
    
        # Continue with normal processing regardless of aggregator status
        logging.info(f"Success: {self.wandb_log_data['goal_reached']}")
        logging.info('\n===================RUN COMPLETE===================\n')
        
        # Generate GIFs if needed
        if self.cfg['log_freq'] == 1:
            create_gif(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'), 
                self.agent.cfg['sensor_cfg']['img_height'], 
                self.agent.cfg['sensor_cfg']['img_width'], 
                agent_cls=self.agent_cls
            )
            create_gif_nav(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )
            create_gif_cvalue(
                os.path.join(os.environ.get("LOG_DIR"), f'{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}'),
                1800, 1800
            )
