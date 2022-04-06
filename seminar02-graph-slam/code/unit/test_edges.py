import unittest
import numpy as np

import transforms as ts
import graph_elements as ge


def test_prior_edge():
    vertex = ge.SE2Vertex(np.array([100.0, 100.0, 0.1]))
    event = {'pose': [500.0, 500.0, 0.1], 'type': 'init'}
    edge = ge.PriorEdge(vertex, event, cov_diag=np.array([0.1, 0.1, 0.1]))
    edge.compute_error()
    assert np.linalg.norm(edge.error - np.array([400.0, 400.0, 0.0]), ord=np.inf) < 1E-5 or \
        np.linalg.norm(edge.error - np.array([-400.0, -400.0, 0.0]), ord=np.inf) < 1E-5
    assert np.linalg.norm(edge.inf.flatten() - np.diag([10.0, 10.0, 10.0]).flatten(), ord=np.inf) < 1E-5


class TestOdometry():
    def test_odometry_forward_right(self):
        R = 10.0
        THETA = np.radians(15)
        T = ts.Transform2D.from_pose([213.0, 0.0, 0.98])
        FINAL_ANGLE_SHIFT = -0.112
        from_pose = (T * ts.Transform2D.from_pose([0.0, 0.0, 0.0])).to_pose()
        from_vertex = ge.SE2Vertex(from_pose)
        to_pose = (T * ts.Transform2D.from_pose([R * np.sin(THETA), R * np.cos(THETA) - R, -THETA + FINAL_ANGLE_SHIFT])).to_pose()
        to_vertex = ge.SE2Vertex(to_pose)
        event = {'command': [10.0, 0.0], 'alpha': [0.001] * 6, 'type': 'control'}
        edge = ge.OdometryEdge(from_vertex=from_vertex, to_vertex=to_vertex, event=event)
        edge.compute_error()
        
        # Prediction should be [2.61799, -0.261799, -0.112]
        # Error should be [7.38201, 0.261799, -0.112]
        assert np.linalg.norm(edge.error - np.array([7.38201, 0.261799, -0.112]), ord=np.inf) < 1E-5
        assert np.linalg.norm(edge.inf.flatten() - np.diag([10.0] * 3).flatten(), ord=np.inf) < 1E-5


    def test_odometry_forward_left(self):
        R = 10.0
        THETA = np.radians(15)
        T = ts.Transform2D.from_pose([213.0, 0.0, 0.98])
        FINAL_ANGLE_SHIFT = -0.112
        from_pose = (T * ts.Transform2D.from_pose([0.0, 0.0, 0.0])).to_pose()
        from_vertex = ge.SE2Vertex(from_pose)
        to_pose = (T * ts.Transform2D.from_pose([R * np.sin(THETA), R - R * np.cos(THETA), THETA + FINAL_ANGLE_SHIFT])).to_pose()
        to_vertex = ge.SE2Vertex(to_pose)
        event = {'command': [10.0, 0.0], 'alpha': [0.001] * 6, 'type': 'control'}
        edge = ge.OdometryEdge(from_vertex=from_vertex, to_vertex=to_vertex, event=event)
        edge.compute_error()
        
        # Prediction should be [2.61799, 0.261799, -0.112]
        # Error should be [7.38201, -0.261799, -0.112]
        assert np.linalg.norm(edge.error - np.array([7.38201, -0.261799, -0.112]), ord=np.inf) < 1E-5
        assert np.linalg.norm(edge.inf.flatten() - np.diag([10.0] * 3).flatten(), ord=np.inf) < 1E-5


    def test_odometry_backward_left(self):
        # Backward left rotation
        R = 10.0
        THETA = np.radians(15)
        T = ts.Transform2D.from_pose([-10.0, 50.0, -0.45])
        FINAL_ANGLE_SHIFT = 0.256
        from_pose = (T * ts.Transform2D.from_pose([0.0, 0.0, 0.0])).to_pose()
        from_vertex = ge.SE2Vertex(from_pose)
        to_pose = (T * ts.Transform2D.from_pose([-R * np.sin(THETA), R - R * np.cos(THETA), -THETA + FINAL_ANGLE_SHIFT])).to_pose()
        to_vertex = ge.SE2Vertex(to_pose)
        event = {'command': [10.0, 0.0], 'alpha': [0.001] * 6, 'type': 'control'}
        edge = ge.OdometryEdge(from_vertex=from_vertex, to_vertex=to_vertex, event=event)
        edge.compute_error()
        
        # Prediction should be [-2.61799, -0.261799, 0.256]
        # Error should be [12.61799, 0.261799, 0.256]
        assert np.linalg.norm(edge.error - np.array([12.61799, 0.261799, 0.256]), ord=np.inf) < 1E-5
        assert np.linalg.norm(edge.inf.flatten() - np.diag([10.0] * 3).flatten(), ord=np.inf) < 1E-5


    def test_odometry_backward_right(self):
        # Backward right rotation
        R = 10.0
        THETA = np.radians(15)
        T = ts.Transform2D.from_pose([213.0, 0.0, 0.98])
        FINAL_ANGLE_SHIFT = 0.45
        from_pose = (T * ts.Transform2D.from_pose([0.0, 0.0, 0.0])).to_pose()
        from_vertex = ge.SE2Vertex(from_pose)
        to_pose = (T * ts.Transform2D.from_pose([-R * np.sin(THETA), R * np.cos(THETA) - R, THETA + FINAL_ANGLE_SHIFT])).to_pose()
        to_vertex = ge.SE2Vertex(to_pose)
        event = {'command': [10.0, 0.0], 'alpha': [0.001] * 6, 'type': 'control'}
        edge = ge.OdometryEdge(from_vertex=from_vertex, to_vertex=to_vertex, event=event)
        edge.compute_error()
        
        # Prediction should be [-2.61799, 0.261799, 0.45]
        # Error should be [12.61799, -0.261799, 0.45]
        assert np.linalg.norm(edge.error - np.array([12.61799, -0.261799, 0.45]), ord=np.inf) < 1E-5
        assert np.linalg.norm(edge.inf.flatten() - np.diag([10.0] * 3).flatten(), ord=np.inf) < 1E-5


    def test_odometry_forward(self):
        # Linear forward
        DISTANCE = 5.44
        T = ts.Transform2D.from_pose([-213.0, 0.5, 0.1])
        FINAL_ANGLE_SHIFT = 0.11
        from_pose = (T * ts.Transform2D.from_pose([0.0, 0.0, 0.0])).to_pose()
        from_vertex = ge.SE2Vertex(from_pose)
        to_pose = (T * ts.Transform2D.from_pose([DISTANCE, 0.0, FINAL_ANGLE_SHIFT])).to_pose()
        to_vertex = ge.SE2Vertex(to_pose)
        event = {'command': [10.0, 0.0], 'alpha': [0.001] * 6, 'type': 'control'}
        edge = ge.OdometryEdge(from_vertex=from_vertex, to_vertex=to_vertex, event=event)
        edge.compute_error()
        
        # Prediction should be [5.44, 0.0, 0.11]
        # Error should be [4.56, 0.0, 0.11]
        assert np.linalg.norm(edge.error - np.array([4.56, 0.0, 0.11]), ord=np.inf) < 1E-5
        assert np.linalg.norm(edge.inf.flatten() - np.diag([10.0] * 3).flatten(), ord=np.inf) < 1E-5


    def test_odometry_backward(self):
        # Linear backward
        DISTANCE = 6.8
        T = ts.Transform2D.from_pose([23.0, -0.5, 0.2])
        FINAL_ANGLE_SHIFT = 0.11
        from_pose = (T * ts.Transform2D.from_pose([0.0, 0.0, 0.0])).to_pose()
        from_vertex = ge.SE2Vertex(from_pose)
        to_pose = (T * ts.Transform2D.from_pose([-DISTANCE, 0.0, FINAL_ANGLE_SHIFT])).to_pose()
        to_vertex = ge.SE2Vertex(to_pose)
        event = {'command': [10.0, 0.0], 'alpha': [0.001] * 6, 'type': 'control'}
        edge = ge.OdometryEdge(from_vertex=from_vertex, to_vertex=to_vertex, event=event)
        edge.compute_error()
        
        # Prediction should be [-6.8, 0.0, 0.11]
        # Error should be [16.8, 0.0, 0.11]
        assert np.linalg.norm(edge.error - np.array([16.8, 0.0, 0.11]), ord=np.inf) < 1E-5
        assert np.linalg.norm(edge.inf.flatten() - np.diag([10.0] * 3).flatten(), ord=np.inf) < 1E-5


def test_landmark_observation_edge():
    pose_vertex = ge.SE2Vertex(np.array([100.0, 100.0, np.radians(90)]))
    feature_vertex = ge.Landmark(np.array([90.0, 105.0]))
    event = {'type': 'point', 'Q': [4.0, 0.0, 0.0, 4.0], 'measurement': [5.0, 10.0]}
    
    edge = ge.LandmarkObservationEdge(pose_vertex=pose_vertex, feature_vertex=feature_vertex, event=event)
    edge.compute_error()
    assert np.linalg.norm(edge.error - np.array([0.0, 0.0]), ord=np.inf) < 1E-5
    assert np.linalg.norm(edge.inf.flatten() - np.diag([0.25, 0.25]).flatten(), ord=np.inf) < 1E-5
