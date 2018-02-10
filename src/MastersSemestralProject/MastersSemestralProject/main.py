import sys
import pygame
import math
import os

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

        img = pygame.image.load(os.path.join("img", "car.jpg"))
        self.car = pygame.transform.scale(img, (self.size.width, self.size.height))

    def rotate_right(self):
        self.angle = self.angle + self.ROTATE_SPEED
        print("right angle=%d", self.angle)

    def rotate_left(self):
        self.angle = self.angle - self.ROTATE_SPEED
        print("left angle=%d", self.angle)

    def forward(self):
        self.speed += 0.25
        if self.speed > self.MAX_FORWARD_SPEED:
            self.speed = self.MAX_FORWARD_SPEED
        print("forward speed=%f", self.speed)

    def backward(self):
        if self.speed > 0:
            self.speed -= 0.5
        else:
            self.speed -= 0.1
        if self.speed < self.MAX_BACKWARD_SPEED:
            self.speed = self.MAX_BACKWARD_SPEED
        print("backward speed=%f", self.speed)

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
        self.position.x += self.speed * math.cos(rad)
        self.position.y += self.speed * math.sin(rad)

    def render(self, screen):
        #car = pygame.surface.Surface((self.size.width, self.size.height), pygame.SRCALPHA)
        #car = pygame.image.load(os.path.join("img", "car.jpg"))
        #car = pygame.transform.scale(car, (self.size.width, self.size.height))
        #car.fill(self.color)
        
        car = pygame.transform.rotate(self.car, -self.angle)
        rect = car.get_rect(center=(self.position.x - self.size.width // 2, self.position.y - self.size.height // 2))
        rect = rect.move(self.size.width // 2, self.size.height // 2)

        screen.blit(car, rect)

        
        #screen.blit(img, rect)
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
    size = [1200, 600]
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Semestral project")

    scene = Scene()
    car = Car(Point2D(100, 100), Size2D(200, 100), 0, CarSensor(5, 135, 10))
    scene.addObject(car)

    quit = False
    clock = pygame.time.Clock()
    while not quit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True

        scene.render(screen)
        pygame.display.flip()

        scene.update()
        clock.tick(30)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main(sys.argv)