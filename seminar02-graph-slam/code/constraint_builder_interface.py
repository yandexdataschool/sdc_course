import numpy as np
import graph_elements as ge

class Constraint(object):
    '''
    Stores constraint information: pose edges and features.
    The content strongly depends on producer.
    '''

    def __init__(self, pose_edges=[], features=[]):
        self.pose_edges = list(pose_edges)
        self.features = list(features)


class IConstraintBuilder(object):
    '''
    Abstract class for constraint builder.
    
    add_event() is populated with sequential events.
    When the class is ready to return new constraint it should return 'True' from ready() method.
    When ready attribute is set build method is invoked to obtain created constraint.
    '''
    
    def add_event(self, event):  
        raise Exception('Not implemented')
    
    def ready(self):
        raise Exception('Not implemented')
    
    def build(self):
        raise Exception('Not implemented')


class PriorEdgeConstraintBuilder(IConstraintBuilder):
    '''
    Builds initialization constraint based on init event
    '''

    def __init__(self, pose_vertices):
        self._pose_vertices = pose_vertices
        self._ready = False
        self._edge = None
    
    def add_event(self, event):
        if event['type'] != 'init':
            self._ready = False
            return
        self._ready = True
        '''
        #########################################
        TO_IMPLEMENT Seminar.Task#2
        '''
    
    def ready(self):
        return self._ready
    
    def build(self):
        if not self.ready():
            return None
        constraint = Constraint(pose_edges=[self._edge])
        self._edge = None
        self._ready = False
        return constraint


class OdometryConstraintBuilder(IConstraintBuilder):
    '''
    Builds odometric constraints based on control measurements
    '''

    def __init__(self, pose_vertices):
        self._pose_vertices = pose_vertices
        self._ready = False
        self._edge = None
    
    def add_event(self, event):
        if event['type'] != 'control':
            self._ready = False
            return
        self._ready = True
        '''
        #########################################
        TO_IMPLEMENT Seminar.Task#4
        '''

    
    def ready(self):
        return self._ready
    
    def build(self):
        if not self.ready():
            return None
        constraint = Constraint(pose_edges=[self._edge])
        self._edge = None
        self._ready = False
        return constraint


class LandmarkConstraintBuilder(IConstraintBuilder):
    def __init__(self, pose_vertices):
        '''
        #########################################
        TO_IMPLEMENT Homework.Task#2
        '''
        self._ready = False
        self._feature = None

    def add_event(self, event):
        '''
        #########################################
        TO_IMPLEMENT Homework.Task#2
        '''
    
    def ready(self):
        return self._ready
    
    def build(self):
        if not self.ready():
            return None
        constraint = Constraint(features=[self._feature])
        self._feature = None
        self._ready = False
        return constraint
