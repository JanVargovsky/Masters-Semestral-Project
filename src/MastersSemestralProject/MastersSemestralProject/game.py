import pygame
import math
import os

debugCollisions = True

class PygameSettings(object):
    width = 1200
    height = 600
    fps = 60
    disableAfterCrash = True

    def get_screen_size():
        return [PygameSettings.width, PygameSettings.height]

class Scene(object):
    background = (0, 0, 0)

    def __init__(self):
        self.track = None
        self.cars = []

    def setTrack(self, track):
        self.track = track

    def addCar(self, car):
        self.cars.append(car)

    def render(self, screen):
        screen.fill(Scene.background)
        self.track.render(screen)
        for car in self.cars:
            car.render(screen)

    def update(self):
        for car in self.cars:
            car.update()

    def checkCollisions(self):
        for car in self.cars:
            car.checkCollisions(self.track)

class GameApp(object):
    def run(self):
        pygame.init()
        screen = pygame.display.set_mode(PygameSettings.get_screen_size())
        pygame.display.set_caption("Semestral project")

        scene = Scene()
        track = RaceTrack()
        scene.setTrack(track)
        car = Car(Point2D(PygameSettings.width / 2, PygameSettings.height / 2), Size2D(50,25), 0, CarSensor(5, 75, 75))
        scene.addCar(car)

        quit = False
        clock = pygame.time.Clock()
        while not quit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit = True

            scene.checkCollisions()
            scene.update()
            scene.render(screen)

            pygame.display.flip()
            clock.tick(PygameSettings.fps)
            pygame.display.set_caption("Semestral project ({0:2.1f})".format(clock.get_fps()))
        pygame.quit()

class Point2D(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Size2D(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

class CarSensor(object):
    color = (255, 255, 0)
    def __init__(self, count, angle, length):
        self.count = count
        self.angle = angle
        self.length = length
        self.sensor = pygame.surface.Surface((length * 2, length * 2), pygame.SRCALPHA)
        self.sensor = self.sensor.convert_alpha()

        start = (length, length)
        offset = -90 + (180 - self.angle) / 2
        for i in range(0, self.count):
            alpha = self.angle / (self.count - 1) * i
            rad = math.radians(alpha)
            end = ((int)(math.cos(rad) * length + length), (int)(math.sin(rad) * length + length))
            print("alpha={0}, pos={1}".format(alpha, end))
            pygame.draw.line(self.sensor, CarSensor.color, start, end)
        if debugCollisions:
            pygame.draw.rect(self.sensor, (255, 0, 0), (0, 0, length * 2, length * 2), 1)

    def render(self, screen, x, y, angle):
        sensor = pygame.transform.rotate(self.sensor, -angle)
        rect = sensor.get_rect(center=(x, y))

        screen.blit(sensor, rect)

class Car(object):
    MAX_FORWARD_SPEED = 5
    MAX_BACKWARD_SPEED = -3
    ROTATE_SPEED = 3

    def __init__(self, position: Point2D, size: Size2D, angle: int, sensor: CarSensor):
        self.position = position
        self.size = size
        self.angle = angle
        self.sensor = sensor
        self.color = (0x0, 0xBF, 0xFF)
        self.speed = 0.0
        self.crash = False
        self.rect = pygame.rect.Rect(position.x, position.y, size.width, size.height)

        car = pygame.image.load(os.path.join("img", "car.jpg"))
        self.car = pygame.transform.scale(car, (self.size.width, self.size.height))

    def rotate_right(self):
        self.angle = self.angle + self.ROTATE_SPEED
        print("right angle={0}".format(self.angle))

    def rotate_left(self):
        self.angle = self.angle - self.ROTATE_SPEED
        print("left angle={0}".format(self.angle))

    def forward(self):
        self.speed += 0.25
        if self.speed > self.MAX_FORWARD_SPEED:
            self.speed = self.MAX_FORWARD_SPEED
        print("forward speed={0}".format(self.speed))

    def backward(self):
        if self.speed > 0:
            self.speed -= 0.5
        else:
            self.speed -= 0.1
        if self.speed < self.MAX_BACKWARD_SPEED:
            self.speed = self.MAX_BACKWARD_SPEED
        print("backward speed={0}".format(self.speed))

    def check_keys(self):
        pressedKeys = pygame.key.get_pressed()
        if pressedKeys[pygame.K_a]:
            self.rotate_left()
        if pressedKeys[pygame.K_d]:
            self.rotate_right()
        if pressedKeys[pygame.K_w]:
            self.forward()
        if pressedKeys[pygame.K_s]:
            self.backward()

    def checkCollisions(self, track):
        if track.intersect(self.rect):
            self.crash = True

    def update(self):
        if PygameSettings.disableAfterCrash and self.crash:
            return

        self.check_keys()

        rad = math.radians(self.angle)
        x = self.position.x + self.speed * math.cos(rad)
        y = self.position.y + self.speed * math.sin(rad)

        #screenRect = pygame.rect.Rect(0,0, Settings.width, Settings.height)
        #carRect = pygame.rect.Rect(x - self.size.width // 2, y - self.size.height // 2, x + self.size.width // 2, y + self.size.height // 2)

        # detect moving out of view
        x = min(x, PygameSettings.width - self.size.width // 2)
        y = min(y, PygameSettings.height - self.size.height // 2)
        x = max(x, self.size.width // 2)
        y = max(y, self.size.height // 2)

        self.position.x = x
        self.position.y = y

    def render(self, screen):
        #car = pygame.surface.Surface((self.size.width, self.size.height), pygame.SRCALPHA)
        #car = pygame.image.load(os.path.join("img", "car.jpg"))
        #car = pygame.transform.scale(car, (self.size.width, self.size.height))
        #car.fill(self.color)
        
        car = pygame.transform.rotate(self.car, -self.angle)
        rect = car.get_rect(center=(self.position.x - self.size.width // 2, self.position.y - self.size.height // 2))
        rect = rect.move(self.size.width // 2, self.size.height // 2)

        screen.blit(car, rect)
        self.rect = car.get_rect(center=(self.position.x, self.position.y))

        if not self.crash:
            rad = math.radians(self.angle)
            sensorX = self.position.x + math.cos(rad) * self.size.width // 2
            sensorY = self.position.y + math.sin(rad) * self.size.width // 2
            self.sensor.render(screen, sensorX, sensorY, self.angle)
        if debugCollisions:
            pygame.draw.rect(screen, (255, 0, 0), rect, 1)


class RaceTrack(object):
    COLOR = (0xFF, 0xFF, 0xFF)

    def __init__(self):
        w = 15
        staticPoints1 = [(0, 50 - w), (100, 50 - w)]
        staticPoints2 = [(0, 50 + w), (100, 50 + w)]

        scaleX = 100. / PygameSettings.width
        scaleY = 100. / PygameSettings.height
        scale = lambda p: (p[0] / scaleX, p[1] / scaleY)
        
        self.points = [list(map(scale, staticPoints1)), list(map(scale, staticPoints2))]

    def render(self, screen):
        pygame.draw.lines(screen, self.COLOR, False, self.points[0])
        pygame.draw.lines(screen, self.COLOR, False, self.points[1])

    #https://www.topcoder.com/community/data-science/data-science-tutorials/geometry-concepts-line-intersection-and-its-applications/
    def linePointsToEquation(self, p1, p2):
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = A * p1[0] + B * p1[1]
        return (A,B,C)

    def isOnLine(self, line, x,y):
        x1 = line[0][0]
        x2 = line[1][0]
        y1 = line[0][1]
        y2 = line[1][1]

        rx = min(x1, x2) <= x and x <= max(x1, x2)
        ry = min(y1, y2) <= y and y <= max(y1, y2)
        return rx and ry

    def lineIntersect(self, line1, line2, selfCall=False):
        A1, B1, C1 = self.linePointsToEquation(line1[0], line1[1])
        A2, B2, C2 = self.linePointsToEquation(line2[0], line2[1])

        det = A1 * B2 - A2 * B1
        if det == 0:
            if selfCall:
                return True
            return self.lineIntersect((line1[0], line2[0]), (line1[1], line2[1]), True)
        else:
            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
            l1 = self.isOnLine(line1, x,y)
            l2 = self.isOnLine(line2, x,y)
            return l1 and l2

    def intersect(self, rect):
        for points in self.points:
            for i in range(1, len(points)):
                line = (points[i - 1], points[i])
                # top
                line2 = (rect.topleft, rect.topright)
                if self.lineIntersect(line, line2):
                    return True
                # bottom
                line2 = (rect.bottomleft, rect.bottomright)
                if self.lineIntersect(line, line2):
                    return True
                #left
                line2 = (rect.bottomleft, rect.topleft)
                if self.lineIntersect(line, line2):
                    return True
                #right
                line2 = (rect.bottomright, rect.topright)
                if self.lineIntersect(line, line2):
                    return True
                
        return False
