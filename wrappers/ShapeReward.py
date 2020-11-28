import numpy as np
import gym

visited = [np.zeros(shape=(21,79))]
class BotWrapper(gym.Wrapper):
    
    def __init__(self,env):
        super().__init__(env)
        self.env = env
        self._actions = self.env._actions
        
        self.previousState = np.ones(shape=(21,79))*32
        self.lastMessage = ""
    def step(self,action : int):
        global visited
        reward = 0
        mask = np.ones(25)
        mask[[-3,-1,0,1]] =0 # carrying capacity and time and start coords
        mask[-2] = -1 # hunger
        mask[12] = 50 # depth

        previousChars = self.env.last_observation[self.env._observation_keys.index('chars')].copy()
        previousStats = self.env.last_observation[self.env._observation_keys.index('blstats')].copy()
        previousDepth = previousStats[12].copy()
        previousCoords = tuple(previousStats[0:2][::-1])
        observations = self.env.step(action)# take the step

        chars = self.env.last_observation[self.env._observation_keys.index('chars')].copy()
        stats = self.env.last_observation[self.env._observation_keys.index('blstats')].copy()
        message = bytes(self.env.last_observation[self.env._observation_keys.index('message')].copy())
        message = message[: message.index(b"\0")].decode("utf-8")
        depth = previousStats[12].copy()
        currentCoords = tuple(stats[0:2][::-1])

        statsReward = np.dot(mask,(stats-previousStats))
        statsReward = statsReward/np.abs(statsReward+1e-9)*(np.abs(statsReward))**0.75
        reward += statsReward #adjust?
        if (currentCoords==previousCoords):
            reward -= 2 # penalize not moving even if it means eating etc
        if (len(visited)<depth):
            visited.append(np.zeros(shape=(21,79)))

        if (visited[depth-1][currentCoords]):
            reward -= 1
        else:
            visited[depth-1][currentCoords] = True
            reward += 1
        if ("You kill" in message):
            reward += 3
        elif ("The door opens" in message):
            reward += 5
        elif ("The door resists" in message):
            reward += 2
        elif ("You hit" in message):
            reward += 1
        elif ("You miss" in message):
            reward += -1.5
        elif ("Ouch" in message):# should already be encompassed
            reward += -1
        newObservations = (observations[0],float(observations[1]+reward),observations[2],observations[3])
        #obs, reward, done, info = env.step(action)
        return newObservations


    def render(self, mode="human"):
        # changing this up a lot

        chars_index = self.env._observation_keys.index("chars")
        if mode == "human":
            message_index = self.env._observation_keys.index("message")
            message = bytes(self.env.last_observation[message_index])


            colors_index = self.env._observation_keys.index("colors")
            chars = self.env.last_observation[chars_index]
            colors = self.env.last_observation[colors_index]
            nh_HE = "\033[0m" #for colouring
            BRIGHT = 8
            chars[np.where(np.logical_or(chars==100,chars==102))] = 46 #remove pets
            colors[np.where(np.logical_or(colors==7,colors ==0))] = 16 #change colors
            chars[np.where(chars==43)] = 93 # replace doors with fullstops or ] 46 or 93
            clearLine = "\033[2K"
            rows,cols = np.where(self.previousState != chars)
            for r,c in zip(rows,cols):
                cursor_pos = "\033[%d;%dH"%(r+1,c)
                start_color = "\033[%d" % bool(colors[r][c] & BRIGHT)
                start_color += ";3%d" % (colors[r][c] & ~BRIGHT)
                start_color += "m"
                
                print(cursor_pos+start_color + chr(chars[r][c]), end=nh_HE)
            
            print("%s%sBluesummit the Stripling    St:18/01 Dx:12 Co:20 In:8 Wi:9 Ch:7  Lawful S:0 "%("\033[23;0H",clearLine),end="")
            print("%s%sDlvl:1  $:0  HP:18(18) Pw:1(1) AC:6  Exp:1 T:0 "%("\033[24;0H",clearLine),end="")
            r,c = np.where(chars==64)
            print("\033[%s;%sH\033[s"%(r[0]+1,c[0]),end="",flush=True)
            print("%s%s"%("\033[0;0H",clearLine)+message[: message.index(b"\0")].decode("utf-8"),end="",flush=True)
            print("\033[u",end="",flush=True)
            self.previousState = chars.copy()
            self.lastMessage = message[: message.index(b"\0")].decode("utf-8")
            if (self.lastMessage==""):
                self.lastMessage = "#"
            return
        elif mode == "ansi":
            chars = self.env.last_observation[chars_index]
            return "\n".join([line.tobytes().decode("utf-8") for line in chars])
        else:
            return super().render(mode=mode)

