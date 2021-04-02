import numpy as np

import transforms as ts

class Edge(object):
    '''
    Base edge class.
    Some methods are abstract and should be redefined.
    Provides numerical Jacobian computations.
    '''

    def __init__(self, vertices):
        self.vertices = vertices
        self._J = None
        self.inf = None
    
    def linearize(self):
        DELTA = 1E-9
        self._J = []
        for vertex in self.vertices:
            J = None
            start_params = vertex.params
            for dim in range(vertex.dim):
                vertex.params = start_params
                delta_params = np.zeros(vertex.dim)
                delta_params[dim] = DELTA
                vertex.update(delta_params)
                self.compute_error()
                error_diff = self.error
                vertex.params = start_params
                delta_params[dim] = -DELTA
                vertex.update(delta_params)
                self.compute_error()
                error_diff -= self.error
                if J is None:
                    J = np.zeros((len(error_diff), vertex.dim))
                J[:, dim] = error_diff / 2.0 / DELTA
            self._J.append(J)

        
    def J(self, vertex_index):
        assert self._J is not None and vertex_index < len(self._J)
        return self._J[vertex_index]
    
    @property
    def inf(self):
        raise Exception('Not implemented')
    
    @property
    def error(self):
        raise Exception('Not implemented')
    
    def compute_error(self):
        raise Exception('Not implemented')
    
    def chi2(self):
        return np.dot(np.dot(self.error, self.inf), self.error)


class Vertex(object):
    '''
    Basic vertex class.
    Update method is abstract and should be reimplemented
    '''

    def __init__(self, params):
        self.params = params
    
    def update(self, delta):
        raise Exception('Not implemented')
    
    @property
    def dim(self):
        return len(self.params)


class SE2Vertex(Vertex):
    '''Vertex class that represents SE(2) class of transformations'''

    def __init__(self, params):
        assert len(params) == 3
        super(SE2Vertex, self).__init__(params)
        
    def update(self, delta):
        transform = np.array([np.cos(self.params[2]), -np.sin(self.params[2]), 0.0,
                              np.sin(self.params[2]), np.cos(self.params[2]), 0.0,
                              0.0, 0.0, 1.0]).reshape((3, 3))
        self.params += np.dot(transform, delta)


class Feature(object):
    '''Represents feature and associated data: feature vertex, related edges and type'''
    
    UNDEFINED = 0
    POINT = 1
    LINE = 2
    
    def __init__(self, vertex, edges, ftype):
        self.vertex = vertex
        self.edges = edges
        self.ftype = ftype

    @property
    def visualization_data(self):
        if self.ftype == Feature.POINT:
            return self.vertex.params
        return None


class PriorEdge(Edge):
    inf = None
    error = None
    '''
    #########################################
    TO_IMPLEMENT Seminar.Task#2
    '''


class OdometryEdge(Edge):
    inf = None
    error = None

    def __init__(self, from_vertex, to_vertex, event):
        super(OdometryEdge, self).__init__([from_vertex, to_vertex])
        alpha = np.array(event['alpha']).reshape((3, 2))
        self.inf = np.diag(1.0 / np.dot(alpha, np.square(np.array(event['command']))))
        self._v, self._w = event['command']
    
    @property
    def from_vertex(self):
        return self.vertices[0]
    
    @property
    def to_vertex(self):
        return self.vertices[1]
    
    def compute_error(self):
        '''
        #########################################
        TO_IMPLEMENT Seminar.Task#3
        '''
        pass


class Landmark(Vertex):
    '''
    Represents positon of feature in the map
    '''
    
    def __init__(self, params):
        assert len(params) == 2
        super(Landmark, self).__init__(params)
    
    def update(self, delta):
        self.params += delta


class LandmarkObservationEdge(Edge):
    inf = None
    error = None

    '''
    #########################################
    TO_IMPLEMENT Homework.Task#1
    '''
