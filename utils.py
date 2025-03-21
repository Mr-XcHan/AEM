import random
import numpy as np
import collections
import torch
import gym
import cv2
from torch import nn
from d3rlpy.dataset import Transition


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.1, std=1)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.1, std=1)
        nn.init.constant_(layer.bias, 0)

class ReplayBuffer:
    """
        Define Replay Buffer.
    """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # deque, FIFO

    def add(self, state, action, reward, next_state, done):  # add one date into buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # sample batch_size data from buffer
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):  # The amount of data in buffer.
        return len(self.buffer)


def calculate_huber_loss(td_error, k=1.0):
    """

    :param td_error:
    :param k:
    :return:
    """
    huber_loss = torch.where(td_error.abs() <= k, 0.5 * td_error.pow(2), k * (td_error.abs() - 0.5 * k))
    return huber_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_timedelta(timedelta):
    total_seconds = int(timedelta)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:0>2d}:{:0>2d}:{:0>2d}'.format(hours, minutes, seconds)


def create_atari_environment(game_name=None, sticky_actions=True, seed=0):
    """Wraps an Atari 2600 Gym environment with some basic preprocessing.

    This preprocessing matches the guidelines proposed in Machado et al. (2017),
    "Revisiting the Arcade Learning Environment: Evaluation Protocols and Open
    Problems for General Agents".

    The created environment is the Gym wrapper around the Arcade Learning
    Environment.

    The main choice available to the user is whether to use sticky actions or not.
    Sticky actions, as prescribed by Machado et al., cause actions to persist
    with some probability (0.25) when a new command is sent to the ALE. This
    can be viewed as introducing a mild form of stochasticity in the environment.
    We use them by default.

    Args:
    game_name: str, the name of the Atari 2600 domain.
    sticky_actions: bool, whether to use sticky_actions as per Machado et al.

    Returns:
    An Atari 2600 environment with some standard preprocessing.
    """
    assert game_name is not None
    game_version = 'v0' if sticky_actions else 'v4'
    full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
    env = gym.make(full_game_name)
    # Strip out the TimeLimit wrapper from Gym, which caps us at 100k frames. We
    # handle this time limit internally instead, which lets us cap at 108k frames
    # (30 minutes). The TimeLimit wrapper also plays poorly with saving and
    # restoring states.
    env.seed(seed)
    env = env.env
    env = AtariPreprocessing(env)
    return env


class AtariPreprocessing(object):
    """A class implementing image preprocessing for Atari 2600 agents.

  Specifically, this provides the following subset from the JAIR paper
  (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

    * Frame skipping (defaults to 4).
    * Terminal signal when a life is lost (off by default).
    * Grayscale and max-pooling of the last two frames.
    * Downsample the screen to a square image (defaults to 84x84).

  More generally, this class follows the preprocessing guidelines set down in
  Machado et al. (2018), "Revisiting the Arcade Learning Environment:
  Evaluation Protocols and Open Problems for General Agents".
  """

    def __init__(self, environment, frame_skip=4, terminal_on_life_loss=False,
                 screen_size=84):
        """Constructor for an Atari 2600 preprocessor.

    Args:
      environment: Gym environment whose observations are preprocessed.
      frame_skip: int, the frequency at which the agent experiences the game.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, size of a resized Atari 2600 frame.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
        if frame_skip <= 0:
            raise ValueError('Frame skip should be strictly positive, got {}'.
                             format(frame_skip))
        if screen_size <= 0:
            raise ValueError('Target screen size should be strictly positive, got {}'.
                             format(screen_size))

        self.environment = environment
        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size

        obs_dims = self.environment.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8)
        ]

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        return gym.spaces.Box(low=0, high=255, shape=(1, self.screen_size, self.screen_size),
                              dtype=np.uint8)

    @property
    def action_space(self):
        return self.environment.action_space

    @property
    def reward_range(self):
        return self.environment.reward_range

    @property
    def metadata(self):
        return self.environment.metadata

    def close(self):
        return self.environment.close()

    def reset(self):
        """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """
        self.environment.reset()
        self.lives = self.environment.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def render(self, mode):
        """Renders the current screen, before preprocessing.

    This calls the Gym API's render() method.

    Args:
      mode: Mode argument for the environment's render() method.
        Valid values (str) are:
          'rgb_array': returns the raw ALE image.
          'human': renders to display via the Gym renderer.

    Returns:
      if mode='rgb_array': numpy array, the most recent screen.
      if mode='human': bool, whether the rendering was successful.
    """
        return self.environment.render(mode)

    def step(self, action):
        """Applies the given action in the environment.

    Remarks:

      * If a terminal state (from life loss or episode end) is reached, this may
        execute fewer than self.frame_skip steps in the environment.
      * Furthermore, in this case the returned observation may not contain valid
        image data and should be ignored.

    Args:
      action: The action to be executed.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      is_terminal: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
        accumulated_reward = 0.

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, info = self.environment.step(action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self.environment.ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            # We max-pool over the last two frames, in grayscale.
            if time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

            if is_terminal:
                break

        # Pool the last two observations.
        observation = self._pool_and_resize()

        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.

    The returned observation is stored in 'output'.

    Args:
      output: numpy array, screen buffer to hold the returned observation.

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
        self.environment.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

    For efficiency, the transformation is done in-place in self.screen_buffer.

    Returns:
      transformed_screen: numpy array, pooled, resized screen.
    """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1],
                       out=self.screen_buffer[0])

        transformed_image = cv2.resize(self.screen_buffer[0],
                                       (self.screen_size, self.screen_size),
                                       interpolation=cv2.INTER_AREA)
        int_image = np.asarray(transformed_image, dtype=np.uint8)
        return np.expand_dims(int_image, axis=2)



