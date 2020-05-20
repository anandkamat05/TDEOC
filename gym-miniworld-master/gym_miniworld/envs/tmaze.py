import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box
from gym import spaces

class TMaze(MiniWorldEnv):
    """
    Two hallways connected in a T-junction
    """

    def __init__(self, **kwargs):
        self.both = False
        self.left=False
        self.right=False

        super().__init__(
            max_episode_steps=280,
            **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        room1 = self.add_rect_room(
            min_x=-1, max_x=8,
            min_z=-2, max_z=2
        )
        room2 = self.add_rect_room(
            min_x=8, max_x=12,
            min_z=-8, max_z=8
        )
        self.connect_rooms(room1, room2, min_z=-2, max_z=2)

        # Add a box at a random end of the hallway
        self.box = Box(color='red')
        if self.both:
            self.box2 = Box(color='red')
            self.place_entity(self.box, room=room2, min_z=room2.max_z - 2, max_z=room2.max_z - 2, min_x=room2.max_x - 2, max_x=room2.max_x - 2)
            self.place_entity(self.box2, room=room2, max_z=room2.min_z + 2, min_z=room2.min_z + 2, max_x=room2.min_x + 2, min_x=room2.min_x + 2)
        elif self.right:
            self.place_entity(self.box, room=room2, min_z=room2.max_z - 2, max_z=room2.max_z - 2, min_x=room2.max_x - 2, max_x=room2.max_x - 2)
        elif self.left:
            self.place_entity(self.box, room=room2, max_z=room2.min_z + 2, min_z=room2.min_z + 2, max_x=room2.min_x + 2, min_x=room2.min_x + 2)
        else:
            if self.rand.bool():
                self.place_entity(self.box, room=room2, min_z=room2.max_z - 2)
            else:
                self.place_entity(self.box, room=room2, max_z=room2.min_z + 2)

        # Choose a random room and position to spawn at
        self.place_agent(
            dir=self.rand.float(-math.pi/4, math.pi/4),
            room=room1
        )

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.both:
            if self.near(self.box) or self.near(self.box2):
                reward = 1
                done = True
        else:
            if self.near(self.box):
                # reward =1
                reward += self._reward()
                done = True

        return obs, reward, done, info

    def set_both(self):
        self.both =True

    def place_left(self):
        if self.right:
            self.right=False
        if self.both:
            self.both=False
        self.left=True

    def place_right(self):
        if self.left:
            self.left=False
        if self.both:
            self.both=False
        self.right=True