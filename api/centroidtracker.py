from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
from math import sqrt
import json

card_directions = {
    "up" : (0,-1),
    "up_right" : (1/sqrt(2),-1/sqrt(2)),
    "right" : (1,0),
    "down_right" : (1/sqrt(2),1/sqrt(2)),
    "down" : (0,1),
    "down_left" : (-1/sqrt(2),1/sqrt(2)),
    "left" : (-1,0),
    "up_left" : (-1/sqrt(2),-1/sqrt(2))
}


class CentroidTracker():
    def __init__(self, maxDisappeared=0, file_name="test.txt"):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.time_stamp = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.objects_direction = OrderedDict()
        self.car_past_direction = dict()
        self.car_initial_position = dict()
        self.road_directions = []
        self.numOfCarExistedLast = dict()
        self.car_initial_time = dict()
        self.entire_hist = dict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.file_name = "temp\\" + file_name
        self.last30_file_name = "temp\\last30" + file_name

        f = open(self.file_name, "w")
        f.close()
        f = open(self.last30_file_name, "w")
        f.close()


    def register(self, centroid, time):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.objects_direction[self.nextObjectID] = 0
        self.disappeared[self.nextObjectID] = 0
        self.car_past_direction[self.nextObjectID] = []
        self.car_initial_position[self.nextObjectID] = centroid
        self.car_initial_time[self.nextObjectID] = time
        self.nextObjectID += 1

    def numberOfObjects(self):
        return self.nextObjectID -1;

    def deregister(self, objectID, time):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries

        if (len(self.car_past_direction[objectID]) > 10): # can safety assume a direction

            x = self.objects[objectID][0] - self.car_initial_position[objectID][0]
            y = self.objects[objectID][1] - self.car_initial_position[objectID][1]
            distance = x**2 + y**2
            speed = distance/(time-self.car_initial_time[objectID])
            average_direction_x = x/(abs(x)+abs(y))
            average_direction_y = y/(abs(x)+abs(y))
            average_direction = (average_direction_x, average_direction_y)
            if (len(self.road_directions) == 0):
                self.road_directions.append((average_direction, 1))
                f = open(self.file_name, "a")
                f.write("time " + str(time) + " " + " x " + str(average_direction[0]) + " y " + str(average_direction[1])+ " speed " + str(speed) + "\n")
                f.close()

                self.numOfCarExistedLast[average_direction] = {
                    'count' : 1,
                    'speed' : [speed]
                }

            else:
                index = 0
                found = False
                for direction in self.road_directions:
                    if (abs(direction[0][0]-average_direction[0]) < 0.3 and abs(direction[0][1]-average_direction[1]) < 0.3):
                        found = True
                        break
                    index+=1

                if (found!=True):
                    self.road_directions.append((average_direction,1))
                    f = open(self.file_name, "a")
                    f.write("time " + str(time) + " " + " x " + str(average_direction[0]) + " y " + str(
                        average_direction[1]) + " speed " + str(speed) + "\n")
                    f.close()
                    self.numOfCarExistedLast[average_direction] = {
                        'count': 1,
                        'speed': [speed]
                    }


                else:
                    self.road_directions[index] = (self.road_directions[index][0], self.road_directions[index][1]+1)
                    f = open(self.file_name, "a")
                    f.write("time " + str(time) + " " + " x " + str(self.road_directions[index][0][0]) + " y " + str(
                        self.road_directions[index][0][1])  + " speed " + str(speed) + "\n")
                    f.close()

                    # _direction = " x " + str(self.road_directions[index][0][0]) + " y " + str(self.road_directions[index][0][1])
                    if self.road_directions[index][0] not in self.numOfCarExistedLast:
                        self.numOfCarExistedLast[self.road_directions[index][0]] = {
                            'count': 0,
                            'speed': []
                        }

                    self.numOfCarExistedLast[self.road_directions[index][0]]['count'] += 1
                    self.numOfCarExistedLast[self.road_directions[index][0]]['speed'].append(speed)

        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.objects_direction[objectID]
        del self.car_past_direction[objectID]

    def update(self, rects, time):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID, time)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects, self.objects_direction, self.road_directions

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], time)

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                # calculate direction
                self.objects_direction[objectID] =  (inputCentroids[col][0] - self.objects[objectID][0], inputCentroids[col][1] - self.objects[objectID][1])

                if (abs(self.objects_direction[objectID][0]) + abs(self.objects_direction[objectID][1]) !=0):
                    normalized_x = self.objects_direction[objectID][0] / (
                            abs(self.objects_direction[objectID][0]) + abs(self.objects_direction[objectID][1]))
                    normalized_y = self.objects_direction[objectID][1] / (
                            abs(self.objects_direction[objectID][0]) + abs(self.objects_direction[objectID][1]))
                    self.car_past_direction[objectID].append((normalized_x, normalized_y))

                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0



                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID, time)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids wderegistere need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], time)

        if (time >= self.time_stamp+30):
            self.time_stamp = self.time_stamp + 30
            # f = open(self.last30_file_name, "a")
            # f.write("time " + str(self.time_stamp)+ "\n")

            last30_dict = {}

            for key, item in card_directions.items():
                last30_dict[key] = {
                    'count': 0,
                    'average_speed': 0
                }

            for key, item in self.numOfCarExistedLast.items():
                min_difference = 1000
                direction = ""
                for card_direction, value in card_directions.items():
                    if (abs(value[0]-key[0])+abs(value[1]-key[1])) < min_difference:
                        min_difference = abs(value[0]-key[0])+abs(value[1]-key[1])
                        direction = card_direction
                
                if direction != "":
                    last30_dict[direction]['count'] += item['count']
                    for speed in item['speed']:
                        last30_dict[direction]['average_speed'] += speed

            for key, item in last30_dict.items():
                if item['count'] >0:
                    # f.write("direction="+ key + " num=" + str(item['count']) + " average_speed=" + str(item['average_speed']/item['count']) + "\n")
                    last30_dict[key]['average_speed'] = last30_dict[key]['average_speed']/item['count']


            self.entire_hist[str(self.time_stamp)] = last30_dict
            with open(self.last30_file_name, 'w') as fp:
                json.dump(self.entire_hist, fp)

            self.numOfCarExistedLast = dict()


            # return the set of trackable objects
        return self.objects, self.objects_direction, self.road_directions
