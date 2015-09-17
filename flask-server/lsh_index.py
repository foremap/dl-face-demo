import numpy as np
import os
import sys
import csv
import random
import time
from collections import defaultdict

from nearpy import Engine
from nearpy.distances import CosineDistance, EuclideanDistance

from nearpy.hashes import RandomBinaryProjections, PCABinaryProjections, HashPermutations, HashPermutationMapper
from nearpy.filters import NearestFilter, UniqueFilter, DistanceThresholdFilter

from scipy.spatial import distance


class LSHSearch:
    def __init__(self, feature_file, dimension, neighbour, lsh_project_num):
        self.feature_file = feature_file
        self.dimension = dimension
        self.neighbour = neighbour
        self.face_feature = defaultdict(str)
        self.ground_truth = defaultdict(int)

        # Create permutations meta-hash
        permutations2 = HashPermutationMapper('permut2')

        tmp_feature = defaultdict(str)
        with open(feature_file, 'rb') as f:
            reader = csv.reader(f, delimiter=' ')
            for name, feature in reader:
                tmp_feature[name] = feature

        matrix = []
        label = []
        for item in tmp_feature.keys():
            v = map(float, tmp_feature[item].split(','))
            matrix.append(np.array(v))
            label.append(item)
        random.shuffle(matrix)
        print 'PCA matric : ', len(matrix)

        rbp_perm2 = PCABinaryProjections('testPCABPHash', lsh_project_num, matrix)
        permutations2.add_child_hash(rbp_perm2)

        # Create engine
        nearest = NearestFilter(self.neighbour)
        self.engine = Engine(self.dimension, lshashes=[permutations2], distance=CosineDistance(), vector_filters=[nearest])

    def build(self):
        with open(self.feature_file, 'rb') as f:
            reader = csv.reader(f, delimiter=' ')
            for name, feature in reader:
                self.face_feature[name] = feature
                person = '_'.join(name.split('_')[:-1])
                self.ground_truth[person] += 1 

        for item in self.face_feature.keys():
            v = map(float, self.face_feature[item].split(','))
            self.engine.store_vector(v, item)

    def query(self, person_feature):
        dists = []
        scores = []

        query = map(float, person_feature.split(','))
        # print '\nNeighbour distances with mutliple binary hashes:'
        # print '  -> Candidate count is %d' % self.engine.candidate_count(query)
        results = self.engine.neighbours(query)
        dists = dists + [x[1] for x in results]
        scores = scores + [x[2] for x in results]

        res = zip(dists, scores)
        res.sort(key = lambda t: t[1])
        return res[:self.neighbour]