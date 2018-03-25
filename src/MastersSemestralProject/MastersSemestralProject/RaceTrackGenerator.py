from math import sqrt

class RaceTrackGenerator(object):
    def __init__(self, w):
        self.w = w

        track1 = ((0,50), (100, 50))
        track2 = ((0, 25), (20,25), (40, 50), (50, 25), (60, 75), (100, 75))
        self.tracks = list()
        self.tracks.append(track1)

    def getCenterTrack(self, id):
        return self.tracks[id]

    def getTrack(self, id):
        def add(u, v):
            return (u[0] + v[0], u[1] + v[1])

        def scaleVector(u, length):
            l = sqrt(u[0] ** 2 + u[1] ** 2)
            v = (u[0] / l * length, u[1] / l * length)
            return v

        def correctPoints(points):
            """points is a collection of points (2 points = line)
            line a = (p1, p2)
            line b = (p3, p4)
            find intersection between line a and b = px
            change a = (p1, px), b = (px, p4)"""
            
            # https://stackoverflow.com/a/20677983/5286021
            def line_intersection(line1, line2):
                xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
                ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

                def det(a, b):
                    return a[0] * b[1] - a[1] * b[0]

                div = det(xdiff, ydiff)
                if div == 0:
                   raise Exception('lines do not intersect')

                d = (det(*line1), det(*line2))
                x = det(d, xdiff) / div
                y = det(d, ydiff) / div
                return x, y

            for i in range(0, len(points) - 2, 2):
                p1, p2, p3, p4 = points[i:i + 4]
                px = line_intersection((p1, p2), (p3, p4))
                points[i + 1] = points[i + 2] = px

            # delete doubled points, points = (p1, p2, p2, p3, p3, p4, p4 ...)
            del points[1:len(points) - 1:2]
            return points

        track = self.getCenterTrack(id)
        points1 = list()
        points2 = list()

        for ta, tb in zip(track, track[1:]):
            # line vector
            u = (tb[0] - ta[0], tb[1] - ta[1])
            # normal vectors (both directions)
            u1 = (-u[1], u[0]) 
            u2 = (u[1], -u[0])

            # scale normal vectors
            su1 = scaleVector(u1, self.w / 2)
            su2 = scaleVector(u2, self.w / 2)

            # shift points
            points1.append(add(ta, su1))
            points1.append(add(tb, su1))

            points2.append(add(ta, su2))
            points2.append(add(tb, su2))

        correctPoints(points1)
        correctPoints(points2)
        return (points1, points2)
