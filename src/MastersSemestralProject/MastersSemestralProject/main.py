import sys
import pygame
import math
import os

class Settings(object):
    width = 1200
    height = 600
    fps = 60

    def get_screen_size():
        return [Settings.width, Settings.height]

class Point2D(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Size2D(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

class CarSensor(object):
    def __init__(self, count, angle, length):
        self.count = count
        self.angle = angle
        self.length = length
        self.sensor = pygame.surface.Surface((length * 2, length * 2), pygame.SRCALPHA)
        self.sensor = self.sensor.convert_alpha()

        color = (255, 0, 0)
        start = (length, length)
        for i in range(0, self.count):
            alpha = self.angle / (self.count - 1) * i
            rad = math.radians(alpha - 90)
            end = ((int)(math.cos(rad) * length + length), (int)(math.sin(rad) * length + length))
            print("alpha={0}, pos={1}".format(alpha, end))
            pygame.draw.line(self.sensor, color, start, end)
        #pygame.draw.rect(self.sensor, (255, 0, 0), (0, 0, length * 2, length *
        #2), 1)

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

    def update(self):
        self.check_keys()

        rad = math.radians(self.angle)
        x = self.position.x + self.speed * math.cos(rad)
        y = self.position.y + self.speed * math.sin(rad)

        #screenRect = pygame.rect.Rect(0,0, Settings.width, Settings.height)
        #carRect = pygame.rect.Rect(x - self.size.width // 2, y - self.size.height // 2, x + self.size.width // 2, y + self.size.height // 2)

        # detect moving out of view
        x = min(x, Settings.width - self.size.width // 2)
        y = min(y, Settings.height - self.size.height // 2)
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

        rad = math.radians(self.angle)
        sensorX = self.position.x + math.cos(rad) * self.size.width // 2
        sensorY = self.position.y + math.sin(rad) * self.size.width // 2
        self.sensor.render(screen, sensorX, sensorY, self.angle)

        #pygame.draw.rect(screen, (255, 0, 0), rect, 1)
class Scene(object):
    def __init__(self):
        self.objects = []

    def addObject(self, object):
        self.objects.append(object)

    def render(self, screen):
        screen.fill((0,0,0))
        for o in self.objects:
            o.render(screen)

    def update(self):
        for o in self.objects:
            o.update()

def main(argv):
    pygame.init()
    screen = pygame.display.set_mode(Settings.get_screen_size())
    pygame.display.set_caption("Semestral project")

    scene = Scene()
    car = Car(Point2D(Settings.width / 2, Settings.height / 2), Size2D(100,50), 0, CarSensor(5, 180, 75))
    scene.addObject(car)

    quit = False
    clock = pygame.time.Clock()
    while not quit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True

        scene.update()
        scene.render(screen)

        pygame.display.flip()
        clock.tick(Settings.fps)
        pygame.display.set_caption("Semestral project ({0:2.1f})".format(clock.get_fps()))

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main(sys.argv)