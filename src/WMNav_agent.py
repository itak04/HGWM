import logging
import math
import random
import habitat_sim
import numpy as np
import cv2
import ast
import concurrent.futures
import json
import re
from typing import Dict, List, Any  # Add missing typing imports

from simWrapper import PolarAction
from utils import *
from api import *
from utils import *
from api import *

class Agent:
    def __init__(self, cfg: dict):
        pass

    def step(self, obs: dict):
        """Primary agent loop to map observations to the agent's action and returns metadata."""
        raise NotImplementedError

    def get_spend(self):
        """Returns the dollar amount spent by the agent on API calls."""
        return 0

    def reset(self):
        """To be called after each episode."""
        pass

class RandomAgent(Agent):
    """Example implementation of a random agent."""
    
    def step(self, obs):
        rotate = random.uniform(-0.2, 0.2)
        forward = random.uniform(0, 1)

        agent_action = PolarAction(forward, rotate)
        metadata = {
            'step_metadata': {'success': 1}, # indicating the VLM succesfully selected an action
            'logging_data': {}, # to be logged in the txt file
            'images': {'color_sensor': obs['color_sensor']} # to be visualized in the GIF
        }
        return agent_action, metadata

class VLMNavAgent(Agent):
    """
    Primary class for the VLMNav agent. Four primary components: navigability, action proposer, projection, and prompting. Runs seperate threads for stopping and preprocessing. This class steps by taking in an observation and returning a PolarAction, along with metadata for logging and visulization.
    """
    explored_color = GREY
    unexplored_color = GREEN
    map_size = 5000
    explore_threshold = 3
    voxel_ray_size = 60
    e_i_scaling = 0.8

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fov = cfg['sensor_cfg']['fov']
        self.resolution = (
            cfg['sensor_cfg']['img_height'],
            cfg['sensor_cfg']['img_width']
        )

        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])       

        assert cfg['navigability_mode'] in ['none', 'depth_estimate', 'segmentation', 'depth_sensor']
        self.depth_estimator = DepthEstimator() if cfg['navigability_mode'] == 'depth_estimate' else None
        self.segmentor = Segmentor() if cfg['navigability_mode'] == 'segmentation' else None
        self.reset()

    def step(self, obs: dict):
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position

        agent_action, metadata = self._choose_action(obs)  # 做完了所有事情
        metadata['step_metadata'].update(self.cfg)

        if metadata['step_metadata']['action_number'] == 0:
            self.turned = self.step_ndx

        # Visualize the chosen action
        chosen_action_image = obs['color_sensor'].copy()
        self._project_onto_image(
            metadata['a_final'], chosen_action_image, agent_state,
            agent_state.sensor_states['color_sensor'], 
            chosen_action=metadata['step_metadata']['action_number'],
            step=self.step_ndx,
            goal=obs['goal']
        )
        metadata['images']['color_sensor_chosen'] = chosen_action_image

        # Visualize the chosen voxel map
        metadata['images']['voxel_map_chosen'] = self._generate_voxel(metadata['a_final'],
                                                                      agent_state=agent_state,
                                                                      chosen_action=metadata['step_metadata']['action_number'],
                                                                      step=self.step_ndx)
        self.step_ndx += 1
        return agent_action, metadata
    
    def get_spend(self):
        return self.actionVLM.get_spend() + self.stoppingVLM.get_spend()

    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.actionVLM.reset()

    def _construct_prompt(self, **kwargs):
        raise NotImplementedError
    
    def _choose_action(self, obs):
        raise NotImplementedError

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.actionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.stoppingVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)

            a_final, images = preprocessing_thread.result()
            called_stop, stopping_response = stopping_thread.result()
        
        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.actionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, stopping_response

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)


        a_final_projected = self._projection(a_final, images, agent_state, obs['goal'])
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state, step=self.step_ndx)
        return a_final_projected, images

    def _stopping_module(self, stopping_images: list[np.array], goal):
        """Determines if the agent should stop."""
        stopping_prompt = self._construct_prompt(goal, 'stopping')
        stopping_response = self.stoppingVLM.call(stopping_images, stopping_prompt)
        dct = self._eval_response(stopping_response)
        if 'done' in dct and int(dct['done']) == 1:
            return True, stopping_response
        
        return False, stopping_response

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        return a_initial

    def _action_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        """Refines the initial set of actions, ensuring spacing and adding a bias towards exploration."""
        min_angle = self.fov/self.cfg['spacing_ratio']
        explore_bias = self.cfg['explore_bias']
        clip_frac = self.cfg['clip_frac']
        clip_mag = self.cfg['max_action_dist']

        explore = explore_bias > 0
        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]
        arrowData = []

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        for theta, mags in unique.items():
            # Reference the map to classify actions as explored or unexplored
            mag = min(mags)
            cart = [self.e_i_scaling*mag*np.sin(theta), 0, -self.e_i_scaling*mag*np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self._global_to_grid(global_coords)
            score = (sum(np.all((topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color), axis=-1)) + 
                    sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
            arrowData.append([clip_frac*mag, theta, score<3])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        filtered.sort(key=lambda x: x[1])
        if filtered == []:
            return []
        if explore:
            # Add unexplored actions with spacing, starting with the longest one
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)
            
                out.append([min(longest[0], clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])
                for i in range(longest_ndx+1, len(f)):
                    if f[i][1] - longest_theta > (min_angle*0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]
                for i in range(longest_ndx-1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle*0.9):
                        
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                for r_i, theta_i, e_i in filtered:
                    if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle*explore_bias:
                        out.append((min(r_i, clip_mag), theta_i, e_i))
                        thetas.add(theta)

        if len(out) == 0:
            # if no explored actions or no explore bias
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2]])
            
            for i in range(longest_ndx+1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx-1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    smallest_theta = filtered[i][1]


        if (out == [] or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            return self._get_default_arrows()
        
        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta, _ in out]

    def _projection(self, a_final: list, images: dict, agent_state: habitat_sim.AgentState, goal: str, candidate_flag: bool=False):
        """
        Projection component of VLMnav. Projects the arrows onto the image, annotating them with action numbers.
        Note actions that are too close together or too close to the boundaries of the image will not get projected.
        """
        a_final_projected = self._project_onto_image(
            a_final, images['color_sensor'], agent_state,
            agent_state.sensor_states['color_sensor'],
            step=self.step_ndx,
            goal=goal,
            candidate_flag=candidate_flag
        )

        if not a_final_projected and (self.step_ndx - self.turned < self.cfg['turn_around_cooldown']) and not candidate_flag:
            logging.info('No actions projected and cannot turn around')
            a_final = self._get_default_arrows()
            a_final_projected = self._project_onto_image(
                a_final, images['color_sensor'], agent_state,
                agent_state.sensor_states['color_sensor'],
                step=self.step_ndx,
                goal=goal
            )

        return a_final_projected

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict):
        """
        Prompting component of VLMNav. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])
        
        response = self.actionVLM.call_chat(prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response

    def _get_navigability_mask(self, rgb_image: np.array, depth_image: np.array, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Get the navigability mask for the current state, according to the configured navigability mode.
        """
        if self.cfg['navigability_mode'] == 'segmentation':
            navigability_mask = self.segmentor.get_navigability_mask(rgb_image)
        else:
            thresh = 1 if self.cfg['navigability_mode'] == 'depth_estimate' else self.cfg['navigability_height_threshold']
            height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
            navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh

        return navigability_mask

    def _get_default_arrows(self):
        """
        Get the action options for when the agent calls stop the first time, or when no navigable actions are found.
        """
        angle = np.deg2rad(self.fov / 2) * 0.7
        
        default_actions = [
            (self.cfg['stopping_action_dist'], -angle),
            (self.cfg['stopping_action_dist'], -angle / 4),
            (self.cfg['stopping_action_dist'], angle / 4),
            (self.cfg['stopping_action_dist'], angle)
        ]
        
        default_actions.sort(key=lambda x: x[1])
        return default_actions

    def _get_radial_distance(self, start_pxl: tuple, theta_i: float, navigability_mask: np.ndarray, 
                             agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, 
                             depth_image: np.ndarray):
        """
        Calculates the distance r_i that the agent can move in the direction theta_i, according to the navigability mask.
        """
        agent_point = [2 * np.sin(theta_i), 0, -2 * np.cos(theta_i)]
        end_pxl = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_pxl is None or end_pxl[1] >= self.resolution[0]:
            return None, None

        H, W = navigability_mask.shape

        # Find intersections of the theoretical line with the image boundaries
        intersections = find_intersections(start_pxl[0], start_pxl[1], end_pxl[0], end_pxl[1], W, H)
        if intersections is None:
            return None, None

        (x1, y1), (x2, y2) = intersections
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        if num_points < 5:
            return None, None
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)

        out = (int(x_coords[-1]), int(y_coords[-1]))
        if not navigability_mask[int(y_coords[0]), int(x_coords[0])]:
            return 0, theta_i

        for i in range(num_points - 4):
            # Trace pixels until they are not navigable
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)]) <= 2:
                out = (x, y)
                break

        if i < 5:
            return 0, theta_i

        if self.cfg['navigability_mode'] == 'segmentation':
            #Simple estimation of distance based on number of pixels
            r_i = 0.0794 * np.exp(0.006590 * i) + 0.616

        else:
            #use depth to get distance
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            camera_coords = unproject_2d(
                *out, depth_image[out[1], out[0]], resolution=self.resolution, focal_length=self.focal_length
            )
            local_coords = global_to_local(
                agent_state.position, agent_state.rotation,
                local_to_global(sensor_state.position, sensor_state.rotation, camera_coords)
            )
            r_i = np.linalg.norm([local_coords[0], local_coords[2]])

        return r_i, theta_i

    def _can_project(self, r_i: float, theta_i: float, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Checks whether the specified polar action can be projected onto the image, i.e., not too close to the boundaries of the image.
        """
        agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]
        end_px = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_px is None:
            return None

        if (
            self.cfg['image_edge_threshold'] * self.resolution[1] <= end_px[0] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[1] and
            self.cfg['image_edge_threshold'] * self.resolution[0] <= end_px[1] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[0]
        ):
            return end_px
        return None

    def _project_onto_image(self, a_final: list, rgb_image: np.ndarray, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, chosen_action: int=None, step: int=None, goal: str='', candidate_flag: bool=False):
        """
        Projects a set of actions onto a single image. Keeps track of action-to-number mapping.
        """
        scale_factor = rgb_image.shape[0] / 1080
        # if candidate_flag:
        #     scale_factor /= 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        circle_color = WHITE
        projected = {}
        if chosen_action == -1:
            put_text_on_image(
                rgb_image, 'TERMINATING EPISODE', text_color=GREEN, text_size=4 * scale_factor,
                location='center', text_thickness=math.ceil(3 * scale_factor), highlight=False
            )
            return projected

        start_px = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        for _, (r_i, theta_i) in enumerate(a_final):
            text_size = 2.4 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)

            end_px = self._can_project(r_i, theta_i, agent_state, sensor_state)
            if end_px is not None:
                action_name = len(projected) + 1
                projected[(r_i, theta_i)] = action_name

                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), RED, math.ceil(5 * scale_factor), tipLength=0.0)
                text = str(action_name)
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

                if chosen_action is not None and action_name == chosen_action:
                    cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
                else:
                    cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
                cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

        if not candidate_flag and ((self.step_ndx - self.turned) >= self.cfg['turn_around_cooldown'] or self.step_ndx == self.turned or (chosen_action == 0)):
            text = '0'
            text_size = 3.1 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

            if chosen_action is not None and chosen_action == 0:
                cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
            cv2.putText(rgb_image, 'TURN AROUND', (text_position[0] // 2, text_position[1] + math.ceil(80 * scale_factor)), font, text_size * 0.75, RED, text_thickness)

        # Add step counter in the top-left corner
        if step is not None:
            step_text = f'step {step}'
            cv2.putText(rgb_image, step_text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Add the target string in the top-right corner
        if goal is not None:
            padding = 20
            text_size = 2.5 * scale_factor
            if candidate_flag:
                text_thickness = 2
            else:
                text_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(f"goal:{goal}", font, text_size, text_thickness)
            text_position = (rgb_image.shape[1] - text_width - padding, padding + text_height)
            cv2.putText(rgb_image, f"goal:{goal}", text_position, font, text_size, (255, 0, 0), text_thickness,
                        cv2.LINE_AA)

        return projected


    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r - 0.5, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)

        # Mark explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _global_to_grid(self, position: np.ndarray, rotation=None):
        """Convert global coordinates to grid coordinates in the agent's voxel map"""
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.voxel_map.shape
        x = int(resolution[1] // 2 + dx * self.scale)
        y = int(resolution[0] // 2 + dz * self.scale)

        if rotation is not None:
            original_coords = np.array([x, y, 1])
            new_coords = np.dot(rotation, original_coords)
            new_x, new_y = new_coords[0], new_coords[1]
            return (int(new_x), int(new_y))

        return (x, y)

    def _generate_voxel(self, a_final: dict, zoom: int=9, agent_state: habitat_sim.AgentState=None, chosen_action: int=None, step: int=None):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        text_size = 1.25
        text_thickness = 1
        rotation_matrix = None
        agent_coords = self._global_to_grid(agent_state.position, rotation=rotation_matrix)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']:
            a_final[(0.75, np.pi)] = 0

        for (r, theta), action in a_final.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt, rotation=rotation_matrix)

            # Draw action arrows and labels
            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), RED, 5, tipLength=0.05)
            text = str(action)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15

            if chosen_action is not None and action == chosen_action:
                cv2.circle(topdown_map, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
            cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)

        # Draw agent's current position
        cv2.circle(topdown_map, agent_coords, radius=15, color=RED, thickness=-1)


        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if step is not None:
            step_text = f'step {step}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def _action_number_to_polar(self, action_number: int, a_final: list):
        """Converts the chosen action number to its PolarAction instance"""
        try:
            action_number = int(action_number)
            if action_number <= len(a_final) and action_number > 0:
                r, theta = a_final[action_number - 1]
                return PolarAction(r, -theta)
            if action_number == 0:
                return PolarAction(0, np.pi)
        except ValueError:
            pass

        logging.info("Bad action number: " + str(action_number))
        return PolarAction.default

    # def _eval_response(self, response: str):
    #     """Converts the VLM response string into a dictionary, if possible"""
    #     try:
    #         eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}') + 1])
    #         if isinstance(eval_resp, dict):
    #             return eval_resp
    #         else:
    #             raise ValueError
    #     except (ValueError, SyntaxError):
    #         logging.error(f'Error parsing response {response}')
    #         return {}
    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        import re
        result = re.sub(r"(?<=[a-zA-Z])'(?=[a-zA-Z])", "\\'", response)
        try:
            eval_resp = ast.literal_eval(result[result.index('{') + 1:result.rindex('}')]) # {{}}
            if isinstance(eval_resp, dict):
                return eval_resp
        except:
            try:
                eval_resp = ast.literal_eval(result[result.rindex('{'):result.rindex('}') + 1]) # {}
                if isinstance(eval_resp, dict):
                    return eval_resp
            except:
                try:
                    eval_resp = ast.literal_eval(result[result.index('{'):result.rindex('}')+1]) # {{}, {}}
                    if isinstance(eval_resp, dict):
                        return eval_resp
                except:
                    logging.error(f'Error parsing response {response}')
                    return {}

class WMNavAgent(VLMNavAgent):
    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.cvalue_map = 10 * np.ones((self.map_size, self.map_size, 3), dtype=np.float16)
        self.goal_position = []
        self.goal_mask = None
        self.panoramic_mask = {}
        self.effective_mask = {}
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.ActionVLM.reset()
        self.PlanVLM.reset()
        self.PredictVLM.reset()
        self.GoalVLM.reset()

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.ActionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.PlanVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.GoalVLM: VLM = vlm_cls(**cfg['model_kwargs'])
        self.PredictVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _update_panoramic_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark circle explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)

        radius = int(np.linalg.norm(np.array(agent_coords) - np.array(point)))
        cv2.circle(self.explored_map, agent_coords, radius, self.explored_color, -1)  # -1表示实心圆

    def _stopping_module(self, obs, threshold_dist=0.8):
        if self.goal_position:
            arr = np.array(self.goal_position)
            # 按列计算平均值
            avg_goal_position = np.mean(arr, axis=0)
            agent_state = obs['agent_state']
            current_position = np.array([agent_state.position[0], agent_state.position[2]])
            goal_position = np.array([avg_goal_position[0], avg_goal_position[2]])
            dist = np.linalg.norm(current_position - goal_position)
            logging.info(f'Current position: {current_position}, Goal position: {goal_position}, Distance: {dist}')
            if dist < threshold_dist:
                return True
        return False


    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        called_stop = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            logging.info('Starting preprocessing and stopping threads')
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, obs)

            a_final, images, a_goal, candidate_images = preprocessing_thread.result() #a_goal is the goal image
            called_stop = stopping_thread.result()

        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none':
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    step=self.step_ndx,
                    goal=obs['goal']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'model': self.ActionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, a_goal, candidate_images

    def _draw_direction_arrow(self, roomtrack_map, direction_vector, position, coords, angle_text='', arrow_length=1):
        # 箭头的终点
        arrow_end = np.array(
            [position[0] + direction_vector[0] * arrow_length, position[1],
             position[2] + direction_vector[2] * arrow_length])  # 假设 y 轴是高度轴，不变

        # 将世界坐标转换为网格坐标
        arrow_end_coords = self._global_to_grid(arrow_end)

        # 绘制箭头
        cv2.arrowedLine(roomtrack_map, (coords[0], coords[1]),
                        (arrow_end_coords[0], arrow_end_coords[1]), WHITE, 4, tipLength=0.1)

        # 绘制文本，表示角度（假设为 30°，你可以根据实际情况调整）
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = 1
        text_thickness = 2

        # 获取文本的宽度和高度，用来居中文本
        (text_width, text_height), _ = cv2.getTextSize(angle_text, font, text_size, text_thickness)

        # 设置文本的位置为箭头终点稍微偏移
        text_end_coords = self._global_to_grid(np.array(
            [position[0] + direction_vector[0] * arrow_length * 1.5, position[1],
             position[2] + direction_vector[2] * arrow_length * 1.5]))
        text_position = (text_end_coords[0] - text_width // 2, text_end_coords[1] + text_height // 2)

        # 绘制文本
        cv2.putText(roomtrack_map, angle_text, text_position, font, text_size, (255, 255, 255), text_thickness,
                    cv2.LINE_AA)

    def generate_voxel(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        # direction vector
        direction_vector = habitat_sim.utils.quat_rotate_vector(agent_state.rotation, habitat_sim.geo.FRONT)

        self._draw_direction_arrow(topdown_map, direction_vector, agent_state.position, agent_coords,
                                   angle_text="0")
        theta_60 = -np.pi / 3
        theta_30 = -np.pi / 6
        y_axis = np.array([0, 1, 0])
        quat_60 = habitat_sim.utils.quat_from_angle_axis(theta_60, y_axis)
        quat_30 = habitat_sim.utils.quat_from_angle_axis(theta_30, y_axis)
        direction_30_vector = habitat_sim.utils.quat_rotate_vector(quat_30, direction_vector)
        self._draw_direction_arrow(topdown_map, direction_30_vector, agent_state.position, agent_coords,
                                   angle_text="30")
        direction_60_vector = direction_30_vector.copy()
        for i in range(5):
            direction_60_vector = habitat_sim.utils.quat_rotate_vector(quat_60, direction_60_vector)
            angle = (i + 1) * 60 + 30
            self._draw_direction_arrow(topdown_map, direction_60_vector, agent_state.position, agent_coords,
                                       angle_text=str(angle))

        text_size = 1.25
        text_thickness = 1
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = str(self.step_ndx)
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
        circle_center = (agent_coords[0], agent_coords[1])
        circle_radius = max(text_width, text_height) // 2 + 15

        cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

        text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
        cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
        cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)


        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (255, 255, 255), 2, cv2.LINE_AA)

        return zoomed_map

    def update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, temp_map: np.ndarray, effective_dist: float=3):
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, 40)

        # Mark directional regions
        cv2.line(temp_map, agent_coords, point, WHITE, 40) # whole area
        unclipped = min(r, effective_dist)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(temp_map, agent_coords, point, GREEN, 40) # effective area

    def _goal_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        min_angle = self.fov / self.cfg['spacing_ratio']

        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]

        arrowData = []

        for theta, mags in unique.items():
            mag = min(mags)
            arrowData.append([mag, theta])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0
        f = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        f.sort(key=lambda x: x[1])
        if f == []:
            return []
        # Add unexplored actions with spacing, starting with the longest one
        if len(f) > 0:
            longest = max(f, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = f.index(longest)

            out.append([longest[0], longest[1]])
            thetas.add(longest[1])
            for i in range(longest_ndx + 1, len(f)):
                if f[i][1] - longest_theta > (min_angle * 0.45):
                    out.append([f[i][0], f[i][1]])
                    thetas.add(f[i][1])
                    longest_theta = f[i][1]
            for i in range(longest_ndx - 1, -1, -1):
                if smallest_theta - f[i][1] > (min_angle * 0.45):
                    out.append([f[i][0], f[i][1]])
                    thetas.add(f[i][1])
                    smallest_theta = f[i][1]

        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta in out]

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        candidate_images = {'color_sensor': obs['color_sensor'].copy()}
        a_goal_projected = None

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)
            if obs['goal_flag']:
                a_goal = self._goal_proposer(a_initial, agent_state)

        a_final_projected = self._projection(a_final, images, agent_state, obs['goal'])
        if obs['goal_flag']:
            a_goal_projected = self._projection(a_goal, candidate_images, agent_state, obs['goal'], candidate_flag=True)
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state, step=self.step_ndx)
        return a_final_projected, images, a_goal_projected, candidate_images

    def navigability(self, obs: dict, direction_idx: int):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range = np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, 120)
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state,
                                                     depth_image)
            if r_i is not None:
                self.update_voxel(
                    r_i, theta_i, agent_state, temp_map
                )
        angle = str(int(direction_idx * 30))
        self.panoramic_mask[angle] = np.all(temp_map == WHITE, axis=-1) | np.all(temp_map == GREEN, axis=-1)
        self.effective_mask[angle] = np.all(temp_map == GREEN, axis=-1)

    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark explored regions
        clipped = min(r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )  # agent的原点在图像坐标系的位置（像素）

        # Generate initial actions
        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        # draw explored circle
        if self.cfg['panoramic_padding'] == True and a_initial:
            r_max, theta_max = max(a_initial, key=lambda x: x[0])
            self._update_panoramic_voxel(r_max, theta_max, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling)
        return a_initial

    def get_spend(self):
        return self.ActionVLM.get_spend()  + self.PlanVLM.get_spend() + self.PredictVLM.get_spend() + self.GoalVLM.get_spend()

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, subtask: str):
        """
        Prompting component of BASEV2. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        prompt_type = 'action'
        action_prompt = self._construct_prompt(goal, prompt_type, subtask, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.ActionVLM.call_chat(prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response) #解析响应获取动作编号
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['ACTION_PROMPT'] = action_prompt
            logging_data['ACTION_RESPONSE'] = response

        return step_metadata, logging_data, response

    def _goal_module(self, goal_image: np.array, a_goal, goal):
        """Determines if the agent should stop."""
        location_prompt = self._construct_prompt(goal, 'goal', num_actions=len(a_goal))
        location_response = self.GoalVLM.call([goal_image], location_prompt)
        dct = self._eval_response(location_response)

        try:
            number = int(dct['Number'])
        except:
            number = None

        return number, location_response

    def _get_goal_position(self, action_goal, idx, agent_state):
        # Initialize with default values
        r, theta = None, None
        
        # Try to find matching action
        for key, value in action_goal.items():
            if value == idx:
                r, theta = key
                break
        
        # Check if we found a match
        if r is None or theta is None:
            # Handle the case where no matching action was found
            logging.warning(f"No action found for index {idx} in action_goal")
            # Return a default position near the agent (1m forward)
            default_position = local_to_global(agent_state.position, agent_state.rotation, [0, 0, -1])
            return default_position, None
        
        # Continue with the original code...
        agent_coords = self._global_to_grid(agent_state.position)
        
        local_goal = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
        global_goal = local_to_global(agent_state.position, agent_state.rotation, local_goal)
        point = self._global_to_grid(global_goal)
        
        # get top down radius
        radius = 1  # real radius (m)
        local_radius = np.array([0, 0, -radius])
        global_radius = local_to_global(agent_state.position, agent_state.rotation, local_radius)
        radius_point = self._global_to_grid(global_radius)
        top_down_radius = int(np.linalg.norm(np.array(agent_coords) - np.array(radius_point)))
        
        temp_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        cv2.circle(temp_map, point, top_down_radius, WHITE, -1)
        goal_mask = np.all(temp_map == WHITE, axis=-1)
        
        return global_goal, goal_mask

    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, a_goal, candidate_images = self._run_threads(obs, [obs['color_sensor']], goal) #并行处理，生成可行动作集合以及检查是否该停止

        goal_image = candidate_images['color_sensor'].copy()
        if a_goal is not None:
            goal_number, location_response = self._goal_module(goal_image, a_goal, goal)
            images['goal_image'] = goal_image
            if goal_number is not None and goal_number != 0:
                goal_position, self.goal_mask = self._get_goal_position(a_goal, goal_number, agent_state)  # get goal position and update goal mask
                self.goal_position.append(goal_position)

        step_metadata['object'] = goal

        # If the model calls stop two times in a row, terminate the episode
        if step_metadata['called_stopping']:  # 如果模型连续两次调用停止，则停止
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if a_goal is not None and goal_number is not None and goal_number != 0:#找到目标
                logging_data = {}
                logging_data['ACTION_NUMBER'] = int(goal_number)
                step_metadata['action_number'] = goal_number
                a_final = a_goal #使用目标导向的动作集
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata, obs['subtask']) #否则使用ActionVLM进行动作选择
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))
        if a_goal is not None:
            logging_data['LOCATOR_RESPONSE'] = location_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images,
            'step': self.step_ndx
        }
        return agent_action, metadata

    def _eval_response(self, response: str, goal: str = None):
        """
        VLM response parser - efficient, robust, with goal awareness.
        
        Args:
            response: The raw VLM response text
            goal: The current navigation goal object (e.g., "bed", "chair")
            
        Returns:
            Parsed dictionary of response data
        """
        import re
        import json
        from typing import Dict, Any
        
        if not response or not isinstance(response, str):
            return self._get_default_response()
        
        # Response preprocessing (remove markdown and extra whitespace)
        response = response.strip()
        markdown_patterns = [
            (r'```json\s*', ''),
            (r'```\s*$', ''),
            (r'^```\s*', ''),
            (r'`{1,3}', ''),
        ]
        
        for pattern, replacement in markdown_patterns:
            response = re.sub(pattern, replacement, response, flags=re.MULTILINE)
        
        response = response.strip()
        
        # Fast path: Try direct JSON parsing first
        if response.startswith('{') and response.endswith('}'):
            try:
                result = json.loads(response)
                if isinstance(result, dict) and result:
                    return self._validate_and_fix_response(result)
            except json.JSONDecodeError:
                pass
        
        # Try intelligent JSON extraction with regex patterns
        json_candidates = self._extract_json_candidates(response)
        for candidate in json_candidates:
            try:
                fixed_candidate = self._fix_common_json_issues(candidate)
                result = json.loads(fixed_candidate)
                if isinstance(result, dict) and result:
                    return self._validate_and_fix_response(result)
            except (json.JSONDecodeError, ValueError):
                continue
        
        # Fallback: Extract key information from text with goal awareness
        logging.info("JSON parsing failed, attempting text-based extraction")
        extracted_info = self._extract_key_info_from_text(response, goal)
        if extracted_info:
            return extracted_info
        
        # Final fallback
        logging.warning(f"Failed to parse VLM response, using default. Response preview: {response[:100]}...")
        return self._get_default_response()
    
    def _extract_json_candidates(self, text: str) -> list:
        """提取可能的JSON候选对象"""
        candidates = []
        
        # 模式1：标准的大括号JSON
        json_pattern = r'\{(?:[^{}]|{[^{}]*})*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        candidates.extend(matches)
        
        # 模式2：查找包含引号的键值对结构
        kv_pattern = r'\{[^{}]*["\'][^"\']*["\'][^{}]*:[^{}]*\}'
        kv_matches = re.findall(kv_pattern, text, re.DOTALL)
        candidates.extend(kv_matches)
        
        # 模式3：处理截断的JSON（尝试补全）
        incomplete_pattern = r'\{[^{}]*["\'][^"\']*["\'][^{}]*:'
        incomplete_matches = re.findall(incomplete_pattern, text)
        for match in incomplete_matches:
            # 尝试简单补全
            if not match.endswith('}'):
                candidates.append(match + ' 1}')  # 假设是数值类型
        
        # 按长度排序，优先尝试更完整的JSON
        return sorted(set(candidates), key=len, reverse=True)
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """修复常见的JSON格式问题"""
        # 替换Python风格的布尔值
        json_str = re.sub(r'\bTrue\b', 'true', json_str)
        json_str = re.sub(r'\bFalse\b', 'false', json_str)
        json_str = re.sub(r'\bNone\b', 'null', json_str)
        
        # 修复单引号
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # 修复尾随逗号
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 修复缺少引号的键
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
        
        return json_str
    
    def _validate_and_fix_response(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """验证并修复响应字典的格式"""
        
        # 对于predicting模块的响应
        if all(k.isdigit() for k in response_dict.keys()):
            fixed_response = {}
            for angle, content in response_dict.items():
                if isinstance(content, dict):
                    # 确保包含必要的字段
                    fixed_content = {
                        'Thought': content.get('Thought', content.get('thought', 'No detailed observation')),
                        'Score': self._extract_score(content)
                    }
                    fixed_response[angle] = fixed_content
                else:
                    # 如果内容不是字典，尝试提取分数
                    score = self._extract_score_from_any(content)
                    fixed_response[angle] = {
                        'Thought': 'Simplified observation',
                        'Score': score
                    }
            return fixed_response
        
        # 对于planning模块的响应
        if 'Flag' in response_dict or 'flag' in response_dict:
            return {
                'Flag': response_dict.get('Flag', response_dict.get('flag', False)),
                'Subtask': response_dict.get('Subtask', response_dict.get('subtask', '{}'))
            }
        
        # 对于action模块的响应
        if 'action' in response_dict or 'Action' in response_dict:
            action_value = response_dict.get('action', response_dict.get('Action', 1))
            return {'action': self._ensure_valid_action(action_value)}
        
        return response_dict
    
    def _extract_score(self, content: Any) -> int:
        """从内容中提取有效的分数"""
        if isinstance(content, dict):
            score = content.get('Score', content.get('score', content.get('value', 1)))
        else:
            score = content
        
        try:
            score_int = int(float(score))
            return max(0, min(10, score_int))  # 限制在0-10范围内
        except (ValueError, TypeError):
            return 1  # 默认分数
    
    def _extract_score_from_any(self, content: Any) -> int:
        """从任意内容中提取分数"""
        if isinstance(content, (int, float)):
            return max(0, min(10, int(content)))
        
        if isinstance(content, str):
            # 查找数字
            numbers = re.findall(r'\b([0-9]|10)\b', content)
            if numbers:
                return int(numbers[-1])  # 取最后一个数字
        
        return 1
    
    def _extract_key_info_from_text(self, text: str, goal: str = None) -> Dict[str, Any]:
        """
        Extract key information from text with dynamic goal-aware processing.
        
        Args:
            text: The response text to analyze
            goal: The current navigation goal object (e.g., "bed", "chair")
            
        Returns:
            Dictionary with extracted information
        """
        if not text:
            return None
            
        text_lower = text.lower()
        goal_lower = goal.lower() if goal else ""
        
        # Planning module response processing (with goal awareness)
        planning_indicators = ['flag', 'subtask', 'goal', 'found']
        if goal_lower:
            planning_indicators.extend([goal_lower, f'standing in front of {goal_lower}', 'very close'])
        
        if any(indicator in text_lower for indicator in planning_indicators):
            # Dynamic negative indicators based on goal
            negative_indicators = [
                'not visible', 'cannot see', "can't see", 'not found', 
                'no visible', 'not in sight', 'not present', 'absent', 'missing',
                'no clear', 'not clearly'
            ]
            
            # Add goal-specific negations
            if goal_lower:
                negative_indicators.extend([
                    f'no {goal_lower}', f'not a {goal_lower}', f'not the {goal_lower}',
                    f'{goal_lower} is not', f'not seeing {goal_lower}', f'didn\'t see {goal_lower}'
                ])
            
            # Check for negative indicators first
            if any(neg in text_lower for neg in negative_indicators):
                logging.info(f"Goal NOT detected due to negative indicators in text: '{text[:100]}...'")
                return {'Flag': False, 'Subtask': {}}
            
            # Only check for positive indicators if no negative ones are found
            strong_goal_indicators = [
                'goal is reached', 'goal found', 'target found',
                'found the', 'clearly visible', 'very close to', 
                'standing in front of', 'reached the', 'arrived at'
            ]
            
            # Add goal-specific positive indicators
            if goal_lower:
                strong_goal_indicators.extend([
                    f'found the {goal_lower}', f'see the {goal_lower}', 
                    f'{goal_lower} is visible', f'{goal_lower} in front of', 
                    f'{goal_lower} clearly visible', f'very close to {goal_lower}', 
                    f'standing in front of {goal_lower}', f'reached the {goal_lower}', 
                    f'arrived at {goal_lower}', f'can see the {goal_lower}',
                    f'there is a {goal_lower}', f'{goal_lower} is here', 
                    f'{goal_lower} is present', f'close to the {goal_lower}'
                ])
            
            # Count matching indicators for stronger evidence requirement
            goal_matches = [indicator for indicator in strong_goal_indicators if indicator in text_lower]
            confidence_threshold = 1  # Require at least this many matches
            
            if goal_matches and len(goal_matches) >= confidence_threshold:
                matched_phrases = ", ".join(goal_matches[:3])
                logging.info(f"✅ GOAL DETECTED: {matched_phrases}")
                print(f"✅ GOAL DETECTED: {matched_phrases}")
                return {'Flag': True, 'Subtask': {}}
            else:
                logging.info(f"Goal not detected in planning text (insufficient evidence)")
                return {'Flag': False, 'Subtask': {}}
        
        # Action selection response processing
        action_patterns = [
            r'action["\s:]*(\d+)',
            r'choose["\s:]*(\d+)',
            r'select["\s:]*(\d+)',
            r'go["\s:]*(\d+)',
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                action_num = int(match.group(1))
                return {'action': self._ensure_valid_action(action_num)}
        
        # Direction-based fallback
        directions = {
            'forward': 1, 'ahead': 1, 'straight': 1,
            'left': 2, 'right': 3, 'back': 0, 'turn': 0
        }
        
        for direction, action in directions.items():
            if direction in text_lower:
                return {'action': action}
        
        return None
    
    def _ensure_valid_action(self, action: Any) -> int:
        """确保动作值有效"""
        try:
            action_int = int(action)
            return max(0, min(10, action_int))  # 限制在合理范围内
        except (ValueError, TypeError):
            return 1
    
    def _get_default_response(self) -> Dict[str, Any]:
        """根据上下文返回默认响应"""
        # 检查当前是哪个模块在调用
        import inspect
        caller_name = inspect.stack()[2].function
        
        if 'predicting' in caller_name:
            # 为predicting模块返回默认的方向评分
            return {
                '30': {'Thought': 'Default exploration direction', 'Score': 5},
                '90': {'Thought': 'Default exploration direction', 'Score': 5},
                '150': {'Thought': 'Default exploration direction', 'Score': 5},
                '210': {'Thought': 'Default exploration direction', 'Score': 5},
                '270': {'Thought': 'Default exploration direction', 'Score': 5},
                '330': {'Thought': 'Default exploration direction', 'Score': 5},
            }
        elif 'planning' in caller_name:
            return {'Flag': False, 'Subtask': '{}'}
        else:
            return {'action': 1}

    def _planning_module(self, planning_image: list[np.array], previous_subtask, goal_reason: str, goal, scene_graph_context=None):
        """Determines if the agent should stop."""
        try:
            planning_prompt = self._construct_prompt(goal, 'planning', previous_subtask, goal_reason, scene_graph_context=scene_graph_context)
            planning_response = self.PlanVLM.call([planning_image], planning_prompt)
            
            # Add debugging info
            print(f"Planning prompt: {planning_prompt[:200]}...")
            print(f"Planning raw response: {planning_response}")
            
            planning_response = planning_response.replace('false', 'False').replace('true', 'True')
            dct = self._eval_response(planning_response, goal)  # Pass the goal parameter
            
            # Validate response format
            if not isinstance(dct, dict) or 'Flag' not in dct or 'Subtask' not in dct:
                print(f"Invalid planning response format: {dct}")
                return {'Flag': False, 'Subtask': '{}'}
                
            return dct
            
        except Exception as e:
            print(f"Planning module error: {e}")
            print(f"Image type: {type(planning_image)}, length: {len(planning_image) if isinstance(planning_image, list) else 'not list'}")
            
            # Return default values
            return {'Flag': False, 'Subtask': '{}'}

    def _predicting_module(self, evaluator_image, goal):
        """Determines if the agent should stop."""
        evaluator_prompt = self._construct_prompt(goal, 'predicting')
        evaluator_response = self.PredictVLM.call([evaluator_image], evaluator_prompt)
        dct = self._eval_response(evaluator_response)

        return dct

    @staticmethod
    def _concat_panoramic(images, angles):
        try:
            height, width = images[0].shape[0], images[0].shape[1]
        except:
            height, width = 480, 640
        background_image = np.zeros((2 * height + 3 * 10, 3 * width + 4 * 10, 3), np.uint8)
        copy_images = np.array(images, dtype=np.uint8)
        for i in range(len(copy_images)): # Iterate through each image
            if i % 2 == 0:
                continue
            copy_images[i] = cv2.putText(copy_images[i], "Angle %d" % angles[i], (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                         2, (255, 0, 0), 6, cv2.LINE_AA)
            row = i // 6
            col = (i // 2) % 3
            background_image[10 * (row + 1) + row * height:10 * (row + 1) + row * height + height:,
            10 * (col + 1) + col * width:10 * (col + 1) + col * width + width, :] = copy_images[i]
        return background_image

    def make_curiosity_value(self, pano_images, goal):
        """
        增强版的好奇心值生成函数，优化思考提取和走廊分析
        """
        angles = (np.arange(len(pano_images))) * 30
        inference_image = self._concat_panoramic(pano_images, angles) # 拼接全景图像用于推理
        
        # 获取方向预测响应
        response = self._predicting_module(inference_image, goal)
        
        explorable_value = {}
        reason = {}
        thoughts = {}
        room_types = {}  # 新增：记录每个方向的房间类型
        spatial_context = {}  # 新增：记录每个方向的空间上下文信息
        
        try:
            # 处理每个方向的评估结果
            for angle, values in response.items():
                if not isinstance(values, dict):
                    continue
                    
                # 提取分数
                explorable_value[angle] = values.get('Score', values.get('score', 5))
                    
                # 智能提取思考内容 - 优先使用Thought，其次使用Explanation
                thought_content = ''
                if 'Thought' in values and values['Thought']:
                    thought_content = values['Thought']
                elif 'Explanation' in values and values['Explanation']:
                    thought_content = values['Explanation']
                
                # 规范化思考内容格式
                if thought_content:
                    # 确保思考内容有适当的结构
                    thoughts[angle] = self._format_direction_thought(thought_content, angle, goal)
                    
                    # 从思考中提取房间类型信息
                    room_types[angle] = self._extract_room_type(thought_content)
                    
                    # 从思考中提取空间上下文
                    spatial_context[angle] = self._extract_spatial_context(thought_content, angle)
                    
                    # 构建推理解释
                    reason[angle] = f"Direction {angle}° reasoning: {thought_content}"
                else:
                    # 默认思考内容
                    thoughts[angle] = f"Direction {angle}: No detailed observation available"
                    reason[angle] = f"Direction {angle}° reasoning: Score {explorable_value[angle]}/10"
        except Exception as e:
            logging.error(f"Error in make_curiosity_value: {e}")
            explorable_value, reason, thoughts = None, None, None
        
        # 存储结构化思考到代理的记忆中
        if thoughts:
            self.direction_thoughts = thoughts
            
            # 新增：存储房间类型和空间上下文
            self.direction_room_types = room_types
            self.direction_spatial_context = spatial_context
            
            # 新增：检测是否处于走廊
            self.in_hallway = self._check_if_in_hallway(thoughts)
            if self.in_hallway:
                print(f"🔍 HALLWAY DETECTION: Agent is in a hallway")
                logging.info(f"HALLWAY DETECTION: Agent is in a hallway")
                
            # 新增：更新全局环境图
            self._update_environment_graph(thoughts, room_types, spatial_context)
        
        return inference_image, explorable_value, reason
    
    def _format_direction_thought(self, thought_content, angle, goal):
        """格式化方向思考内容，确保包含必要的结构信息"""
        # 如果思考内容太短，添加更多结构
        if len(thought_content) < 10:
            return f"Direction {angle}: Limited visibility with score {5}/10"
            
        # 确保包含对目标物体的引用
        if goal.lower() not in thought_content.lower():
            thought_content += f". No {goal} visible in this direction."
            
        # 检测并标记房间类型
        room_indicators = ["bedroom", "bathroom", "kitchen", "living room", "hallway", "dining room", "office"]
        has_room_type = any(room in thought_content.lower() for room in room_indicators)
        
        if not has_room_type:
            thought_content += ". Room type unclear."
            
        return thought_content
    
    def _extract_room_type(self, thought_content):
        """从思考内容中提取房间类型"""
        thought_lower = thought_content.lower()
        
        room_mapping = {
            "bedroom": ["bedroom", "bed room", "master bedroom"],
            "bathroom": ["bathroom", "bath", "toilet", "shower"],
            "kitchen": ["kitchen", "cooking", "stove", "fridge", "refrigerator"],
            "living room": ["living room", "lounge", "sofa", "couch", "tv area"],
            "dining room": ["dining room", "dining area", "dining table"],
            "hallway": ["hallway", "corridor", "passage", "passageway", "hall"],
            "office": ["office", "study", "work", "desk"]
        }
        
        for room_type, indicators in room_mapping.items():
            if any(indicator in thought_lower for indicator in indicators):
                return room_type
                
        return "unknown"
    
    def _extract_spatial_context(self, thought_content, angle):
        """从思考内容中提取空间上下文信息"""
        spatial_indicators = {
            "open door": ["door", "doorway", "entrance", "open door"],
            "stairway": ["stair", "staircase", "stairway"],
            "wall": ["wall", "dead end", "no passage"],
            "window": ["window", "glass"],
            "connected space": ["connect", "lead to", "path to", "access to"],
            "new area": ["new area", "unexplored", "different room"]
        }
        
        context = []
        thought_lower = thought_content.lower()
        
        for context_type, indicators in spatial_indicators.items():
            if any(indicator in thought_lower for indicator in indicators):
                context.append(context_type)
                
        # 添加方向上下文
        context.append(f"angle_{angle}")
        
        return context
    
    def _check_if_in_hallway(self, thoughts):
        """检查代理是否处于走廊中"""
        hallway_indicators = [
            "hallway", "corridor", "passageway", "passage", 
            "multiple doors", "several rooms", "connecting", 
            "multiple directions", "junction"
        ]
        
        # 检查是否有多个方向提到走廊特征
        hallway_directions = 0
        for angle, thought in thoughts.items():
            if any(indicator in thought.lower() for indicator in hallway_indicators):
                hallway_directions += 1
                
        # 如果至少两个方向提到走廊特征，认为代理位于走廊
        return hallway_directions >= 2
    
    def _update_environment_graph(self, thoughts, room_types, spatial_context):
        """基于方向思考更新环境图结构"""
        if not hasattr(self, 'environment_graph'):
            self.environment_graph = {
                'rooms': set(),
                'connections': {},
                'objects': {},
                'visited': set()
            }
            
        # 更新已识别的房间
        for angle, room_type in room_types.items():
            if room_type != "unknown":
                self.environment_graph['rooms'].add(room_type)
                
                # 记录房间关系
                if "hallway" in room_types.values():
                    hallway_angle = [a for a, r in room_types.items() if r == "hallway"][0]
                    if room_type != "hallway":
                        self._add_connection("hallway", room_type, hallway_angle, angle)
                        
        # 标记当前位置为已访问
        if hasattr(self, 'in_hallway') and self.in_hallway:
            self.environment_graph['visited'].add("hallway")

    @staticmethod
    def _merge_evalue(arr, num):
        return np.minimum(arr, num)

    def update_curiosity_value(self, explorable_value, reason):
        try:
            final_score = {}
            for i in range(12):
                if i % 2 == 0:
                    continue
                last_angle = str(int((i-2)*30)) if i != 1 else '330'
                angle = str(int(i * 30))
                next_angle = str(int((i+2)*30)) if i != 11 else '30'
                if np.all(self.panoramic_mask[angle] == False):
                    continue
                intersection1 = self.effective_mask[last_angle] & self.effective_mask[angle]
                intersection2 = self.effective_mask[angle] & self.effective_mask[next_angle]

                mask_minus_intersection = self.effective_mask[angle] & ~intersection1 & ~intersection2

                self.cvalue_map[mask_minus_intersection] = self._merge_evalue(self.cvalue_map[mask_minus_intersection], explorable_value[angle]) # update explorable value map
                if np.all(intersection2 == False):
                    continue
                self.cvalue_map[intersection2] = self._merge_evalue(self.cvalue_map[intersection2], (explorable_value[angle] + explorable_value[next_angle]) / 2)  # update explorable value map
            if self.goal_mask is not None:
                               self.cvalue_map[self.goal_mask] = 10.0
            for i in range(12):
                if i % 2 == 0:
                    continue
                angle = str(int(i * 30))
                if np.all(self.panoramic_mask[angle] == False):
                    final_score[i] = explorable_value[angle]
                else:
                    final_score[i] = np.mean(self.cvalue_map[self.panoramic_mask[angle]])

            idx = max(final_score, key=final_score.get)
            final_reason = reason[str(int(idx * 30))]
        except:
            idx = np.random.randint(0, 12)
            final_reason = ''
        return idx, final_reason

    def draw_cvalue_map(self, agent_state: habitat_sim.AgentState=None, zoom: int=9):
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        cvalue_map = (self.cvalue_map / 10 * 255).astype(np.uint8)


        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Zoom the map
        max_x, max_y = cvalue_map.shape[1], cvalue_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = cvalue_map[y1:y2, x1:x2]

        if self.step_ndx is not None:
            step_text = f'step {self.step_ndx}'
            cv2.putText(zoomed_map, step_text, (30, 90), font, 3, (0, 0, 0), 2, cv2.LINE_AA)

        return zoomed_map

    def make_plan(self, pano_images, previous_subtask, goal_reason, goal):
        response = self._planning_module(pano_images, previous_subtask, goal_reason, goal)

        try:
            goal_flag, subtask = response['Flag'], response['Subtask']
        except:
            print("planning failed!")
            print('response:', response)
            goal_flag, subtask = False, '{}'

        return goal_flag, subtask


    def _construct_prompt(self, goal: str, prompt_type:str, subtask: str='{}', reason: str='{}', num_actions: int=0):
        if prompt_type == 'goal':
            location_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
            f"There are {num_actions} red arrows superimposed onto your observation, which represent potential positions. " 
            f"These are labeled with a number in a white circle, which represent the location you can move to. "
            f"First, tell me whether the {goal} is in the image, and make sure the object you see is ACTUALLY a {goal}, return number 0 if if there is no {goal}, or if you are not sure. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. "
            f'Second, if there is {goal} in the image, then determine which circle best represents the location of the {goal}(close enough to the target. If a person is standing in that position, they can easily touch the {goal}), and give the number and a reason. '
            f'If none of the circles represent the position of the {goal}, return number 0, and give a reason why you returned 0. '
            "Format your answer in the json {{'Number': <The number you choose>}}")
            return location_prompt
        if prompt_type == 'predicting':
            evaluator_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the panoramic image describing your surrounding environment, each image contains a label indicating the relative rotation angle(30, 90, 150, 210, 270, 330) with red fonts. "
            f'Your job is to assign a score to each direction (ranging from 0 to 10), judging whether this direction is worth exploring. The following criteria should be used: '
            f'To help you describe the layout of your surrounding,  please follow my step-by-step instructions: '
            f'(1) If there is no visible way to move to other areas and it is clear that the target is not in sight, assign a score of 0. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
            f'(2) If the {goal} is found, assign a score of 10.  ' 
            f'(3) If there is a way to move to another area, assign a score based on your estimate of the likelihood of finding a {goal}, using your common sense. Moving to another area means there is a turn in the corner, an open door, a hallway, etc. Note you CANNOT GO THROUGH CLOSED DOORS. CLOSED DOORS and GOING UP OR DOWN STAIRS are not considered. '
            "For each direction, provide an explanation for your assigned score. Format your answer in the json {'30': {'Score': <The score(from 0 to 10) of angle 30>, 'Explanation': <An explanation for your assigned score.>}, '90': {...}, '150': {...}, '210': {...}, '270': {...}, '330': {...}}. "
            "Answer Example: {'30': {'Score': 0, 'Explanation': 'Dead end with a recliner. No sign of a bed or any other room.'}, '90': {'Score': 2, 'Explanation': 'Dining area. It is possible there is a doorway leading to other rooms, but bedrooms are less likely to be directly adjacent to dining areas.'}, ..., '330': {'Score': 2, 'Explanation': 'Living room area with a recliner.  Similar to 270, there is a possibility of other rooms, but no strong indication of a bedroom.'}}")
            return evaluator_prompt
        if prompt_type == 'planning':
            if reason != '' and subtask != '{}':
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you the following elements:"
                f"(1)<The observed image>: The image taken from its current location. "
                f"(2){reason}. This explains why you should go in this direction. "
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found and the previous subtask {subtask} has not completed, continue to complete the last subtask {subtask} that has not been completed.'
                f'(3) If the {goal} is not found and the previous subtask {subtask} has already been completed. Identify a new subtask by describing where you are going next to be more likely to find clues to the the {goal} and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            else:
                planning_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location."
                f'Your job is to describe next place to go. '
                f'To help you plan your best next step, I can give you some human suggestions:. '
                f'(1) If the {goal} appears in the image, directly choose the target as the next step in the plan. Note a chair must have a backrest and a chair is not a stool. Note a chair is NOT sofa(couch) which is NOT a bed. '
                f'(2) If the {goal} is not found, describe where you are going next to be more likely to find clues to the the {goal} and analyze the room type and think about whether the {goal} is likely to occur in that direction. Note you need to pay special attention to open doors and hallways, as they can lead to other unseen rooms. Note GOING UP OR DOWN STAIRS is an option. '
                "Format your answer in the json {{'Subtask': <Where you are going next>, 'Flag': <Whether the target is in your view, True or False>}}. "
                "Answer Example: {{'Subtask': 'Go to the hallway', 'Flag': False}} or {{'Subtask': "+f"'Go to the {goal}'"+", 'Flag': True}} or {{'Subtask': 'Go to the open door', 'Flag': True}}")
            return planning_prompt
        if prompt_type == 'action':
            if subtask != '{}':
                action_prompt = (
                f"TASK: {subtask}. Your final task is to NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                f"In order to complete the subtask {subtask} and eventually the final task NAVIGATING TO THE NEAREST {goal.upper()}. Explain which action acheives that best. "
                "Return your answer as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            else:
                action_prompt = (
                    f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                    f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. "
                    f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                    f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                    "Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
                )
            return action_prompt

        raise ValueError('Prompt type must be goal, predicting, planning, or action')