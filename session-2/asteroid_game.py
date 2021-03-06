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
    ], dtype=object)
    return T

def scaleMatrix(sx, sy):
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])
    return S

def dot(X, Y):
    # Check if passed array is pure python or numpy.ndarray
    if type(X) == list:
        X = np.array(X)
    if type(Y) == list:
        Y = np.array(Y)

    # Check if X is vector
    if len(X.shape) == 1:
        X = np.expand_dims(X, axis=0)   # Change shape from (n,) to (1,n)

    # Check if Y is vector
    if len(Y.shape) == 1:
        Y = np.expand_dims(Y, axis=1)   # Change shape from (n,) to (n,1)

    # shape of dot product is X rows and Y columns
    dot_product = np.zeros((X.shape[0], Y.shape[1]), dtype='f')

    for x in range(X.shape[0]):  # iterate over X rows
        for y in range(Y.shape[1]):  # iterate over Y columns
            # multiply according elements from both matrix/vectors and sum them up
            dot_product[x][y] = sum(X[x][k] * Y[k][y] for k in range(Y.shape[0]))

    # Remove added axis from shape
    return np.squeeze(dot_product)

a = [
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]
]
b = np.array([3, 2, 1])
print(dot(a, b))


def vec_2dto3d(vec2):
    I = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    vec3 = dot(I, vec2) + np.array([0, 0, 1])
    return vec3

def vec_3dto2d(vec3):
    I = np.array([
        [1, 0, 0],
        [0, 1, 0],
    ])
    vec2 = dot(I, vec3)
    return vec2


class Character:
    def __init__(self, pos=[0, 0], scale=[1, 1]):
        self.__angle = 0

        self.geometry = []
        self.pos = np.array(pos)
        self.color = 'g'
        self.x_data = []
        self.y_data = []

        self.C = np.identity(3)
        self.R = rotMatrix(self.__angle)
        self.S = scaleMatrix(sx=scale[0], sy=scale[1])
        self.T = translateMatrix(dx=self.pos[0], dy=self.pos[1])
        self.C = self.T
        self.C = dot(self.C, self.R)
        self.C = dot(self.C, self.S)

        self.direction = np.array([0.0, 1.0])
        self.speed = np.random.uniform(0.05, 0.1, 1)
    
    def setAngle(self, angle):
        self.__angle = angle
        self.R = rotMatrix(self.__angle)
        self.direction = np.array([
            self.R[0][1],
            self.R[0][0]
        ])

        self.C = translateMatrix(self.pos[0], self.pos[1])
        self.C = dot(self.C, self.R)
        self.C = dot(self.C, self.S)
        self.C = dot(self.C, translateMatrix(0, -0.333))  # Centre of mass for player is ~ -0.333

    def calculateCenterOfMass(self):
        x_sum = 0
        y_sum = 0
        for i in range(0, len(self.geometry) - 1):
            x_sum += self.x_data[i]
            y_sum += self.y_data[i]
        x_c = x_sum / (len(self.geometry) - 1)
        y_c = y_sum / (len(self.geometry) - 1)
        self.pos = np.array([x_c, y_c])

    def getAngle(self):
        return self.__angle

    def getCurPos(self):
        return self.pos

    def move(self):
        self.x_data = []
        self.y_data = []

        # Update geometry co-ordinates
        for vec2 in self.geometry:
            vec3 = vec_2dto3d(np.array(vec2))
            vec3_r = dot(self.C, vec3)
            vec2_ = vec_3dto2d(vec3_r)

            self.x_data.append(vec2_[0])
            self.y_data.append(vec2_[1])

        # Calculate centre of mass using average coordinates
        self.calculateCenterOfMass()

        # move character in each axis according to direction
        self.pos[0] += self.direction[0] * self.speed
        self.pos[1] += self.direction[1] * self.speed
        self.T = translateMatrix(dx=0, dy=self.speed)
        self.C = dot(self.C, self.T)

    def draw(self):
        plt.plot(self.x_data, self.y_data, c=self.color)


class Bullet(Character):
    def __init__(self, start_pos, direction, trans_matrix, scale=[1, 1]):
        super().__init__(start_pos, scale)
        self.C = trans_matrix
        self.speed = 0.5
        self.generateGeometry()

    def generateGeometry(self):
        self.geometry = np.array([
            [-0.1, 0],
            [0.1, 0],
            [0, 0.1],
            [-0.1, 0]
        ])


class Player(Character):
    def __init__(self, start_pos, scale=[1, 1]):
        super().__init__(start_pos, scale)
        self.generateGeometry()

    def getDirection(self):
        return self.direction

    def getTransMatrix(self):
        return self.C

    def generateGeometry(self):
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],  # Tip coordinates for the ship
            [-1, 0]
        ])
        

class Asteroid(Character):
    def __init__(self, start_pos, radius, angle, scale=[1, 1]):
        super().__init__(start_pos, scale)
        self.n = 20
        self.checks = 0
        self.r = radius
        self.color = (np.random.random(), np.random.random(), np.random.random())

        self.generateGeometry()
        self.setAngle(angle)

    def getGeometry(self):
        return self.geometry

    def getRadius(self):
        return self.r

    # generate points using sin and cos functions and add some distortion to the lines
    def generateGeometry(self):
        for x in range(0, self.n + 1):
            random_noise = np.random.uniform(low=0.1, high=0.3)
            x_point = np.cos(2 * np.pi/self.n * x) * self.r + random_noise
            y_point = np.sin(2 * np.pi/self.n * x) * self.r
            self.geometry.append([x_point, y_point])
    
    def checkOutOfBounds(self):
        # Keep track of recent change number to avoid asteroid getting stuck outside the boundary
        self.checks += 1
        if (self.pos[0] >= 9 or self.pos[0] <= -9) and self.checks > 10:
            self.speed = -self.speed
            self.checks = 0
        elif (self.pos[1] >= 9 or self.pos[1] <= -9) and self.checks > 10:
            self.speed = -self.speed
            self.checks = 0


characters = []
player = Player(start_pos=[0.0, 0.0], scale=[0.5, 1])
characters.append(player)

asteroid1 = Asteroid(start_pos=[3.0, 3.0], radius=1.2, angle=25)
asteroid2 = Asteroid(start_pos=[-2.0, 3.0], radius=0.75, angle=-56)
asteroid3 = Asteroid(start_pos=[6.0, -5.0], radius=0.5, angle=12)
asteroid4 = Asteroid(start_pos=[7.0, 2.0], radius=1, angle=-120)
asteroid5 = Asteroid(start_pos=[4.0, 4.0], radius=0.5, angle=-180)
asteroid6 = Asteroid(start_pos=[1.0, 5.0], radius=0.5, angle=150)
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
    elif event.key == 'v':
        start_pos = player.getCurPos()
        trans_matrix = player.getTransMatrix()
        direction = player.getDirection()
        bullet = Bullet(start_pos, direction, trans_matrix, scale=[0.5, 1])
        characters.append(bullet)
    elif event.key == 'left':
        player.setAngle(player.getAngle() + 10)
    elif event.key == 'right':
        player.setAngle(player.getAngle() - 10)

fig, _ = plt.subplots()
fig.canvas.mpl_connect('key_press_event', handleEvents)


def checkOutOfRangeBullet(character):
    bullet_pos = character.getCurPos()
    if bullet_pos[0] >= 10 \
            or bullet_pos[0] <= -10 \
            or bullet_pos[1] >= 10 \
            or bullet_pos[1] <= -10:
        characters.remove(character)

def checkAsteroidHit(character):
    bullet_pos = character.getCurPos()
    # Iterate over to check if asteroid is hit
    for character_ in characters:
        if isinstance(character_, Asteroid):
            asteroid_pos = character_.getCurPos()
            asteroid_radius = character_.getRadius()

            # Check if bullet is in asteroid's collision box
            x_positive_bound = bullet_pos[0] <= (asteroid_pos[0] + asteroid_radius)
            x_negative_bound = bullet_pos[0] >= (asteroid_pos[0] - asteroid_radius)
            y_positive_bound = bullet_pos[1] <= (asteroid_pos[1] + asteroid_radius)
            y_negative_bound = bullet_pos[1] >= (asteroid_pos[1] - asteroid_radius)
            # If true then remove both asteroid and bullet
            if (x_positive_bound and x_negative_bound) and (y_positive_bound and y_negative_bound):
                characters.remove(character_)
                characters.remove(character)


while is_running:
    plt.clf()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    num_asteroids = 0

    for character in characters:
        if isinstance(character, Bullet):
            checkOutOfRangeBullet(character)
            checkAsteroidHit(character)

        # Bounce back asteroid
        if isinstance(character, Asteroid):
            character.checkOutOfBounds()

        # Move every character one update forward
        character.move()

        # Draw everything on plot
        character.draw()

        # Display score
        if isinstance(character, Asteroid):
            num_asteroids += 1

    plt.title(f"Asteroids left: {num_asteroids}")

    if num_asteroids == 0:
        plt.title(f"Game finished!")
        plt.pause(2)
        is_running = False
        plt.close('all')
        continue

    plt.draw()
    plt.pause(1e-3)