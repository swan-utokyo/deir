"""
This file is used to define and register custom MiniGrid environments.
The code was created based on the official implementation of gym-minigrid
(version 1.0.3) and may differ from the latest version.
"""

from gym_minigrid.envs import DoorKeyEnv
from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, Door, Key, Wall, COLOR_NAMES, DIR_TO_VEC, Ball, Box
from gym_minigrid.register import register
from gym_minigrid.roomgrid import RoomGrid


class CustomDoorKeyEnv(MiniGridEnv):

    def __init__(self, size=8, agent_view_size=7, max_steps=None, disable_penalty=False):
        super().__init__(
            grid_size=size,
            max_steps=10 * size * size if max_steps is None else max_steps,
            agent_view_size=agent_view_size,
        )
        self.disable_penalty = disable_penalty

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Create a vertical splitting wall
        splitIdx = self._rand_int(2, width - 2)
        self.grid.vert_wall(splitIdx, 0)

        # Place the agent at a random position and orientation
        # on the left side of the splitting wall
        self.place_agent(size=(splitIdx, height))

        # Place a door in the wall
        doorIdx = self._rand_int(1, width - 2)
        self.put_obj(Door('yellow', is_locked=True), splitIdx, doorIdx)

        # Place a yellow key on the left side
        self.place_obj(
            obj=Key('yellow'),
            top=(0, 0),
            size=(splitIdx, height)
        )

        self.mission = "use the key to open the door and then get to the goal"

    def _reward(self):
        if self.disable_penalty:
            return 1
        return 1 - 0.9 * (self.step_count / self.max_steps)


class DoorKeyEnv8x8ViewSize9x9(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=8, agent_view_size=9)

class DoorKeyEnv8x8ViewSize5x5(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=8, agent_view_size=5)

class DoorKeyEnv8x8ViewSize3x3(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=8, agent_view_size=3)

class DoorKeyEnv16x16ViewSize9x9(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16, agent_view_size=9)

class DoorKeyEnv16x16ViewSize5x5(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16, agent_view_size=5)

class DoorKeyEnv16x16ViewSize3x3(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=16, agent_view_size=3)

class DoorKeyEnv32x32ViewSize9x9(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=32, agent_view_size=9)

class DoorKeyEnv32x32ViewSize5x5(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=32, agent_view_size=5)

class DoorKeyEnv32x32ViewSize3x3(CustomDoorKeyEnv):
    def __init__(self):
        super().__init__(size=32, agent_view_size=3)

class DoorKeyEnv32x32(DoorKeyEnv):
    def __init__(self):
        super().__init__(size=32)

register(
    id='MiniGrid-DoorKey-8x8-ViewSize-9x9-v0',
    entry_point='src.env.minigrid_envs:DoorKeyEnv8x8ViewSize9x9'
)
register(
    id='MiniGrid-DoorKey-8x8-ViewSize-5x5-v0',
    entry_point='src.env.minigrid_envs:DoorKeyEnv8x8ViewSize5x5'
)
register(
    id='MiniGrid-DoorKey-8x8-ViewSize-3x3-v0',
    entry_point='src.env.minigrid_envs:DoorKeyEnv8x8ViewSize3x3'
)
register(
    id='MiniGrid-DoorKey-16x16-ViewSize-9x9-v0',
    entry_point='src.env.minigrid_envs:DoorKeyEnv16x16ViewSize9x9'
)
register(
    id='MiniGrid-DoorKey-16x16-ViewSize-5x5-v0',
    entry_point='src.env.minigrid_envs:DoorKeyEnv16x16ViewSize5x5'
)
register(
    id='MiniGrid-DoorKey-32x32-v0',
    entry_point='src.env.minigrid_envs:DoorKeyEnv32x32'
)


class CustomKeyCorridor(RoomGrid):
    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        seed=None,
        agent_view_size=7,
    ):
        self.obj_type = obj_type

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=30*room_size**2,
            seed=seed,
            agent_view_size=agent_view_size,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.obj = obj
        self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                done = True

        return obs, reward, done, info


class KeyCorridorS6R3V5(CustomKeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            num_rows=3,
            seed=seed,
            agent_view_size=5,
        )
class KeyCorridorS6R3V3(CustomKeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=6,
            num_rows=3,
            seed=seed,
            agent_view_size=3,
        )
class KeyCorridorS8R4(CustomKeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=8,
            num_rows=4,
            seed=seed,
            agent_view_size=7,
        )
class KeyCorridorS10R5(CustomKeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=10,
            num_rows=5,
            seed=seed,
            agent_view_size=7,
        )
class KeyCorridorS12R6(CustomKeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=12,
            num_rows=6,
            seed=seed,
            agent_view_size=7,
        )


register(
    id='MiniGrid-KeyCorridorS6R3V5-v0',
    entry_point='src.env.minigrid_envs:KeyCorridorS6R3V5'
)
register(
    id='MiniGrid-KeyCorridorS6R3V3-v0',
    entry_point='src.env.minigrid_envs:KeyCorridorS6R3V3'
)
register(
    id='MiniGrid-KeyCorridorS8R4-v0',
    entry_point='src.env.minigrid_envs:KeyCorridorS8R4'
)
register(
    id='MiniGrid-KeyCorridorS10R5-v0',
    entry_point='src.env.minigrid_envs:KeyCorridorS10R5'
)
register(
    id='MiniGrid-KeyCorridorS12R6-v0',
    entry_point='src.env.minigrid_envs:KeyCorridorS12R6'
)


class CustomFourRooms(MiniGridEnv):

    def __init__(self, agent_pos=None, goal_pos=None, agent_view_size=5):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=19, max_steps=100, agent_view_size=agent_view_size)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info

class FourRoomsViewSize5x5(CustomFourRooms):
    def __init__(self):
        super().__init__(agent_view_size=5)

class FourRoomsViewSize3x3(CustomFourRooms):
    def __init__(self):
        super().__init__(agent_view_size=3)

register(
    id="MiniGrid-FourRooms-ViewSize-5x5-v0",
    entry_point="src.env.minigrid_envs:FourRoomsViewSize5x5"
)
register(
    id="MiniGrid-FourRooms-ViewSize-3x3-v0",
    entry_point="src.env.minigrid_envs:FourRoomsViewSize3x3"
)


class MultiRoom:
    def __init__(self,
        top,
        size,
        entryDoorPos,
        exitDoorPos
    ):
        self.top = top
        self.size = size
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos

class CustomMultiRoomEnv(MiniGridEnv):
    def __init__(self,
        minNumRooms,
        maxNumRooms,
        maxRoomSize=10,
        agent_view_size=7,
        grid_size=25,
        max_steps=None,
        disable_penalty=False,
    ):
        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize
        self.disable_penalty = disable_penalty

        self.rooms = []

        super(CustomMultiRoomEnv, self).__init__(
            grid_size=grid_size,
            max_steps=self.maxNumRooms * 20 if max_steps is None else max_steps,
            agent_view_size=agent_view_size
        )

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms+1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx-1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Place the final goal in the last room
        self.goal_pos = self.place_obj(Goal(), roomList[-1].top, roomList[-1].size)

        self.mission = 'traverse the rooms to get to the goal'

    def _placeRoom(
        self,
        numLeft,
        roomList,
        minSz,
        maxSz,
        entryDoorWall,
        entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz+1)
        sizeY = self._rand_int(minSz, maxSz+1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(MultiRoom(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    def _reward(self):
        if self.disable_penalty:
            return 1
        return 1 - 0.9 * (self.step_count / self.max_steps)

class MultiRoomEnvN12(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=7,
        )
class MultiRoomEnvN12V3(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=3,
        )
class MultiRoomEnvN12MaxSteps1k(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=7,
            max_steps=1000,
        )
class MultiRoomEnvN12MaxSteps1kV3(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=3,
            max_steps=1000,
        )
class MultiRoomEnvN12MaxSteps2k(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=7,
            max_steps=2000,
        )
class MultiRoomEnvN12MaxSteps2kV3(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=3,
            max_steps=2000,
        )
class MultiRoomEnvN12MaxSteps3k(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=7,
            max_steps=3000,
        )
class MultiRoomEnvN12MaxSteps3kV3(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=3,
            max_steps=3000,
        )
class MultiRoomEnvN12MaxSteps500(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=7,
            max_steps=500,
        )
class MultiRoomEnvN12MaxSteps500V3(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=3,
            max_steps=500,
        )
class MultiRoomEnvN12MaxSteps600(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=7,
            max_steps=600,
        )
class MultiRoomEnvN12MaxSteps600V3(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=3,
            max_steps=600,
        )
class MultiRoomEnvN12MaxSteps800(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=7,
            max_steps=800,
        )
class MultiRoomEnvN12MaxSteps800V3(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=12,
            maxNumRooms=12,
            agent_view_size=3,
            max_steps=800,
        )
class MultiRoomEnvN30MaxSteps1k(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=1000,
        )
class MultiRoomEnvN30MaxSteps2k(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=2000,
        )
class MultiRoomEnvN30MaxSteps3k(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=3000,
        )
class MultiRoomEnvN30VS3MS1k(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=3,
            max_steps=1000,
        )
class MultiRoomEnvN30VS3MS2k(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=3,
            max_steps=2000,
        )
class MultiRoomEnvN30VS3MS3k(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=3,
            max_steps=3000,
        )
class MultiRoomEnvN30MS100NP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=100,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS300NP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=300,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS500NP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=500,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS1kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=1000,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS2kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=2000,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS3kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=3000,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS4kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=4000,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS5kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=5000,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS6kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=6000,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS7kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=7000,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS8kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=8000,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS9kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=9000,
            disable_penalty=True,
        )
class MultiRoomEnvN30MS10kNP(CustomMultiRoomEnv):
    def __init__(self):
        super().__init__(
            minNumRooms=30,
            maxNumRooms=30,
            grid_size=45,
            agent_view_size=7,
            max_steps=10000,
            disable_penalty=True,
        )

register(
    id='MiniGrid-MultiRoom-N12-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12'
)
register(
    id='MiniGrid-MultiRoom-N12-ViewSize-3x3-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12V3'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps1k-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps1k'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps1k-ViewSize-3x3-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps1kV3'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps2k-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps2k'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps2k-ViewSize-3x3-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps2kV3'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps3k-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps3k'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps3k-ViewSize-3x3-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps3kV3'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps500-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps500'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps500-ViewSize-3x3-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps500V3'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps600-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps600'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps600-ViewSize-3x3-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps600V3'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps800-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps800'
)
register(
    id='MiniGrid-MultiRoom-N12-MaxSteps800-ViewSize-3x3-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN12MaxSteps800V3'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps1k-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MaxSteps1k'
)
register(
    id='MiniGrid-MultiRoom-N30-ViewSize3x3-MaxSteps1k-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30VS3MS1k'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps2k-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MaxSteps2k'
)
register(
    id='MiniGrid-MultiRoom-N30-ViewSize3x3-MaxSteps2k-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30VS3MS2k'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps3k-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MaxSteps3k'
)
register(
    id='MiniGrid-MultiRoom-N30-ViewSize3x3-MaxSteps3k-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30VS3MS3k'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps100-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS100NP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps300-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS300NP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps500-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS500NP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps1k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS1kNP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps2k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS2kNP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps3k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS3kNP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps4k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS4kNP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps5k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS5kNP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps6k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS6kNP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps7k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS7kNP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps8k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS8kNP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps9k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS9kNP'
)
register(
    id='MiniGrid-MultiRoom-N30-MaxSteps10k-NoPenalty-v0',
    entry_point='src.env.minigrid_envs:MultiRoomEnvN30MS10kNP'
)


class CustomObstructedMazeEnv(RoomGrid):
    def __init__(self,
                 num_rows,
                 num_cols,
                 num_rooms_visited,
                 seed=None,
                 agent_view_size=7,
                 ):
        room_size = 6
        max_steps = 4 * num_rooms_visited * room_size ** 2

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_cols,
            max_steps=max_steps,
            seed=seed,
            agent_view_size=agent_view_size,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Define all possible colors for doors
        self.door_colors = self._rand_subset(COLOR_NAMES, len(COLOR_NAMES))
        # Define the color of the ball to pick up
        self.ball_to_find_color = COLOR_NAMES[0]
        # Define the color of the balls that obstruct doors
        self.blocking_ball_color = COLOR_NAMES[1]
        # Define the color of boxes in which keys are hidden
        self.box_color = COLOR_NAMES[2]

        self.mission = "pick up the %s ball" % self.ball_to_find_color

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                done = True

        return obs, reward, done, info

    def add_door(self, i, j, door_idx=0, color=None, locked=False, key_in_box=False, blocked=False):
        """
        Add a door. If the door must be locked, it also adds the key.
        If the key must be hidden, it is put in a box. If the door must
        be obstructed, it adds a ball in front of the door.
        """
        door, door_pos = super().add_door(i, j, door_idx, color, locked=locked)

        if blocked:
            vec = DIR_TO_VEC[door_idx]
            blocking_ball = Ball(self.blocking_ball_color) if blocked else None
            self.grid.set(door_pos[0] - vec[0], door_pos[1] - vec[1], blocking_ball)

        if locked:
            obj = Key(door.color)
            if key_in_box:
                box = Box(self.box_color) if key_in_box else None
                box.contains = obj
                obj = box
            self.place_in_room(i, j, obj)

        return door, door_pos

class ObstructedMaze_Full_V3(CustomObstructedMazeEnv):
    """
    A blue ball is hidden in one of the 4 corners of a 3x3 maze. Doors
    are locked, doors are obstructed by a ball and keys are hidden in
    boxes.
    """

    def __init__(self, agent_room=(1, 1), key_in_box=True, blocked=True,
                 num_quarters=4, num_rooms_visited=25, seed=None, agent_view_size=3):
        self.agent_room = agent_room
        self.key_in_box = key_in_box
        self.blocked = blocked
        self.num_quarters = num_quarters

        super().__init__(
            num_rows=3,
            num_cols=3,
            num_rooms_visited=num_rooms_visited,
            seed=seed,
            agent_view_size=agent_view_size,
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        middle_room = (1, 1)
        # Define positions of "side rooms" i.e. rooms that are neither
        # corners nor the center.
        side_rooms = [(2, 1), (1, 2), (0, 1), (1, 0)][:self.num_quarters]
        for i in range(len(side_rooms)):
            side_room = side_rooms[i]

            # Add a door between the center room and the side room
            self.add_door(*middle_room, door_idx=i, color=self.door_colors[i], locked=False)

            for k in [-1, 1]:
                # Add a door to each side of the side room
                self.add_door(*side_room, locked=True,
                              door_idx=(i+k)%4,
                              color=self.door_colors[(i+k)%len(self.door_colors)],
                              key_in_box=self.key_in_box,
                              blocked=self.blocked)

        corners = [(2, 0), (2, 2), (0, 2), (0, 0)][:self.num_quarters]
        ball_room = self._rand_elem(corners)

        self.obj, _ = self.add_object(*ball_room, "ball", color=self.ball_to_find_color)
        self.place_agent(*self.agent_room)

register(
    id="MiniGrid-ObstructedMaze-Full-V3-v0",
    entry_point="src.env.minigrid_envs:ObstructedMaze_Full_V3"
)