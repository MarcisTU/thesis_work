import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 10)
plt.ion()

#####
# Matrix and vector operations
###
def rotMatrix(degrees):
    theta = np.radians(degrees)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R

def translateMatrix(dx, dy):
    T = np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])
    return T

def scaleMatrix(sx, sy):
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    return S

def dot(X, Y):
    # shape of dot product result is X rows and Y columns
    result = np.zeros((X.shape[0], Y.shape[1]))

    for i in range(X.shape[0]):  # iterate over X rows
        for j in range(Y.shape[1]):  # iterate over Y columns
            # multiply according elements from both matrix/vectors and sum them up
            result[i][j] = sum(X[i][k] * Y[k][j] for k in range(Y.shape[0]))

    return result

def vec_2dto3d(vec2):
    I = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    vec3 = dot(I, vec2) + np.array([[0], [0], [1]])
    return vec3

def vec_3dto2d(vec3):
    I = np.array([
        [1, 0, 0],
        [0, 1, 0],
    ])
    vec2 = dot(I, vec3)
    return vec2

#######

class Character:
    def __init__(self, pos):
        self.__angle = 0

        self.geometry = []
        self.color = 'g'
        self.s = np.array([0.5, 1])

        self.C = np.identity(3)
        self.R = rotMatrix(self.__angle)
        self.S = scaleMatrix(sx=self.s[0], sy=self.s[1])
        self.T = translateMatrix(dx=5, dy=0)

        self.pos = np.array(pos)
        self.dirInit = np.array([0.0, 1.0])
        self.dir = np.array(self.dirInit)
        self.speed = np.random.uniform(0.1, 0.3, 1)
    
    def setAngle(self, angle):
        self.__angle = angle
        self.R = rotMatrix(self.__angle)
        # keep track of direction angle
        self.dir = np.array([
            [self.R[0][1]],
            [self.R[0][0]]
        ])

    def getAngle(self):
        return self.__angle

    def draw(self):
        x_data = []
        y_data = []
        
        # move 0.1 in forward direction every frame
        # move ship in each axis according to rotation
        d_x = self.dir[0] * self.speed
        d_y = self.dir[1] * self.speed
        self.pos[0] += d_x
        self.pos[1] += d_y
        self.T = translateMatrix(dx=self.pos[0], dy=self.pos[1])

        # rescale to (1:1 ratio) to avoid skew effect when rotating
        diff = np.abs(self.s[0] - self.s[1])
        if self.s[0] > self.s[1]:
            sy = self.s[1] + diff
            sx = self.s[0]
        elif self.s[1] > self.s[0]:
            sy = self.s[1]
            sx = self.s[0] + diff
        else:
            sy = self.s[1]
            sx = self.s[0]
        self.S = scaleMatrix(sx, sy)

        # chain dot products
        self.C = self.S
        self.C = dot(self.C, self.T)
        self.C = dot(self.C, self.R)

        # Scale back to desired scale
        self.S = scaleMatrix(sx=self.s[0], sy=self.s[1])
        self.C = dot(self.C, self.S)

        for vec2 in self.geometry:
            # reshape the dimension to have a valid shape for dot. (2, 1) instead of (2,)
            vec2_resh = np.array(vec2).reshape((2,1))

            vec3 = vec_2dto3d(vec2_resh)
            vec3_r = dot(self.C, vec3)
            vec2_ = vec_3dto2d(vec3_r)    
        
            x_data.append(vec2_[0])
            y_data.append(vec2_[1])

        plt.plot(x_data, y_data, c=self.color)


class Player(Character):
    def __init__(self, start_pos):
        super().__init__(start_pos)

        self.generateGeometry()
    
    def generateGeometry(self):
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])
        

class Asteroid(Character):
    def __init__(self, start_pos, radius, angle):
        super().__init__(start_pos)
        # Number of points to draw for circle
        self.n = 20
        self.r = radius
        self.color = (np.random.random(), np.random.random(), np.random.random())

        self.generateGeometry()
        self.setAngle(angle)

    # generate points using sin and cos functins and add some distortion to the lines
    def generateGeometry(self):
        self.geometry = [
            (
                self.r * 4 + (np.cos(2 * np.pi/self.n * x) * self.r * 2 + np.random.rand(1)[0]),
                self.r * 2 + (np.sin(2 * np.pi/self.n * x) * self.r)
            ) for x in range(0, self.n+1)
        ]
    
    def checkOutOfBounds(self):
        if self.pos[0] > 9.5 or self.pos[0] < -9.5:
            self.dir[0] = -self.dir[0]
        elif self.pos[1] > 9.5 or self.pos[1] < -9.5:
            self.dir[1] = -self.dir[1]


#####
# Create player and Asteroids
###
characters = []
player = Player([0.0, 0.0]) 
characters.append(player)

asteroid1 = Asteroid([3.0, 3.0], radius=1.2, angle=25)
asteroid2 = Asteroid([-2.0, 3.0], radius=0.75, angle=-56)
asteroid3 = Asteroid([6.0, -5.0], radius=0.5, angle=12)
asteroid4 = Asteroid([7.0, 2.0], radius=1, angle=-120)
asteroid5 = Asteroid([4.0, 4.0], radius=0.5, angle=-180)
asteroid6 = Asteroid([1.0, 5.0], radius=0.5, angle=150)
characters.append(asteroid1)
characters.append(asteroid2)
characters.append(asteroid3)
characters.append(asteroid4)
characters.append(asteroid5)
characters.append(asteroid6)

is_running = True

def handleEvents(event):
    global is_running, player
    if event.key == 'escape':
        is_running = False
        plt.close('all')
    elif event.key == 'left':
        player.setAngle(player.getAngle() + 5)
    elif event.key == 'right':
        player.setAngle(player.getAngle() - 5)

fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', handleEvents)

while is_running:
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    for character in characters:
        character.draw()
        if isinstance(character, Player):
            plt.title(f"angle: {character.getAngle()}")
        else:
            character.checkOutOfBounds()
    
    plt.draw()
    plt.pause(1e-3)