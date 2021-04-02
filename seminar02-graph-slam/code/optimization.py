import inspect
import itertools
import operator
import sys

import numpy as np

import graph_elements as ge
import constraint_builder_interface as cbi

class Optimization(object):
    '''
    Represents optimization system
    
    At the initialization it should create all vertices, edges and features
    (that is meta entity for related feature vertices and edges). optimize() method linearize all edges,
    construct linear system and solve it via LM methods.
    It's the backend and most mathematically involed part of assignment. It's OK not to fully understand it.
    '''
 

    def __init__(self, timeline):
        self._init_pose_vertices(timeline)
        self._init_constraints(timeline)
        self._Hpp = None
        self._Hpl = None
        self._Hll = None
        self._bp = None
        self._bl = None
        self._lambda_value = None
        self._lambda_factor = 2.0


    def _init_pose_vertices(self, timeline):
        self._pose_vertices = []
        '''
        #########################################
        TO_IMPLEMENT Seminar.Task#1
        '''
        pass


    def _init_constraints(self, timeline):
        builder_classes = []
        print('name:', __name__)
        for name, obj in inspect.getmembers(cbi):
            if not inspect.isclass(obj):
                continue
            if not issubclass(obj, cbi.IConstraintBuilder):
                continue
            if name == 'IConstraintBuilder':
                continue
            print('builder names:', name)
            builder_classes.append(obj)
        builders = [obj(self._pose_vertices) for obj in builder_classes]
        self._features = []
        self._pose_edges = []
        for events_per_frame in timeline:
            for event in events_per_frame:
                for builder in builders:
                    builder.add_event(event)
                    if not builder.ready():
                        continue
                    constraint = builder.build()
                    self._pose_edges.extend(constraint.pose_edges)
                    self._features.extend(constraint.features)
        self._features = list(set(self._features))
    
    def _linearize(self):
        list(map(lambda edge: edge.linearize(), self._all_edges))
        
    @property
    def _all_edges(self):
        return self._pose_edges + sum(map(lambda feature: feature.edges, self._features), [])
    
    def _assemble(self):
        pose_var_dims = np.concatenate(
            [[0], np.cumsum([vertex.dim for vertex in self._pose_vertices])],
            axis=0)
        feature_var_dims = np.concatenate(
            [[0], np.cumsum([feature.vertex.dim for feature in self._features])],
            axis=0)
        self._Hpp = np.zeros((pose_var_dims[-1], pose_var_dims[-1]))
        self._bp = np.zeros(pose_var_dims[-1])
        pose_vertex_to_index = dict(zip(self._pose_vertices, range(len(self._pose_vertices))))
        for edge in self._pose_edges:
            vertex_indices = range(len(edge.vertices))
            local_to_global = lambda vertex_index: pose_vertex_to_index[edge.vertices[vertex_index]]
            var_range = lambda vertex_index: (pose_var_dims[local_to_global(vertex_index)],
                                              pose_var_dims[local_to_global(vertex_index) + 1])
            for vertex_index_1, vertex_index_2 in itertools.product(vertex_indices, vertex_indices):
                vertex_1_range = var_range(vertex_index_1)
                vertex_2_range = var_range(vertex_index_2)
                J1 = edge.J(vertex_index_1)
                J2 = edge.J(vertex_index_2)
                self._Hpp[vertex_1_range[0]:vertex_1_range[1], vertex_2_range[0]:vertex_2_range[1]] += \
                    np.dot(np.dot(J1.T, edge.inf), J2)
            for i in range(len(edge.vertices)):
                vertex_range = var_range(i)
                J = edge.J(i)
                self._bp[vertex_range[0]:vertex_range[1]] += np.dot(np.dot(J.T, edge.inf), edge.error)
        if len(self._features) == 0:
            return
        self._Hll = np.zeros((feature_var_dims[-1], feature_var_dims[-1]))
        self._Hpl = np.zeros((pose_var_dims[-1], feature_var_dims[-1]))
        self._bl = np.zeros(feature_var_dims[-1])
        feature_vertex_to_index = dict(
            zip([feature.vertex for feature in self._features], range(len(self._features))))
        for feature_index, feature in enumerate(self._features):
            feature_vertex_global_index = feature_vertex_to_index[feature.vertex]
            feature_var_range = (feature_var_dims[feature_vertex_global_index],
                                 feature_var_dims[feature_vertex_global_index + 1])
            for edge in feature.edges:
                assert edge.feature_vertex == feature.vertex
                pose_vertex_global_index = pose_vertex_to_index[edge.pose_vertex]
                pose_var_range = (pose_var_dims[pose_vertex_global_index],
                                  pose_var_dims[pose_vertex_global_index + 1])
                assert edge.vertices.index(edge.pose_vertex) == 0
                Jp = edge.J(0)
                Jf = edge.J(1)
                self._Hpp[pose_var_range[0]:pose_var_range[1], pose_var_range[0]:pose_var_range[1]] += \
                    np.dot(np.dot(Jp.T, edge.inf), Jp)
                self._Hll[feature_var_range[0]:feature_var_range[1], feature_var_range[0]:feature_var_range[1]] += \
                    np.dot(np.dot(Jf.T, edge.inf), Jf)
                self._Hpl[pose_var_range[0]:pose_var_range[1], feature_var_range[0]:feature_var_range[1]] += \
                    np.dot(np.dot(Jp.T, edge.inf), Jf)
                self._bp[pose_var_range[0]:pose_var_range[1]] += np.dot(np.dot(Jp.T, edge.inf), edge.error)
                self._bl[feature_var_range[0]:feature_var_range[1]] += np.dot(np.dot(Jf.T, edge.inf), edge.error)
    
    def _reduce_solve(self):
        if len(self._features) == 0:
            return np.linalg.solve(self._Hpp, -self._bp)
        Hll_inv = np.zeros_like(self._Hll)
        shift = 0
        for feature in self._features:
            dim = feature.vertex.dim
            Hll_inv[shift:(shift + dim), shift:(shift + dim)] = \
                np.linalg.inv(self._Hll[shift:(shift + dim), shift:(shift + dim)])
            shift += dim
        H_reduced = self._Hpp - np.dot(np.dot(self._Hpl, Hll_inv), self._Hpl.T)
        b_reduced = self._bp - np.dot(np.dot(self._Hpl, Hll_inv), self._bl)
        return np.linalg.solve(H_reduced, -b_reduced)
    
    def _restore_solve(self, params_update):
        b = self._bl + np.dot(self._Hpl.T, params_update)
        return np.linalg.solve(self._Hll, -b)

    def _solve_and_update(self):
        if len(self._pose_edges) == 0:
            return
        if self._lambda_value is None:
            self._lambda_value = self._Hpp.diagonal().max()
            if 0 < len(self._features):
                self._lambda_value = max(self._lambda_value, self._Hll.diagonal().max())
            SCALE_FACTOR = 1E-5
            self._lambda_value *= SCALE_FACTOR
        self.compute_errors()
        chi2_before = self.get_chi2()
        levenberg_iter = 0
        rho = -1.0
        if 0 < len(self._features):
            b_combined = np.concatenate([self._bp, self._bl], axis=0)
        else:
            b_combined = self._bp
        LEVENBERG_ITERATIONS = 10
        while rho < 0.0 and levenberg_iter < LEVENBERG_ITERATIONS:
            self._Hpp += self._lambda_value * np.eye(*self._Hpp.shape)
            if 0 < len(self._features):
                self._Hll += self._lambda_value * np.eye(*self._Hll.shape)
            pose_params_update = self._reduce_solve()
            if 0 < len(self._features):
                feature_params_update = self._restore_solve(pose_params_update)
            else:
                feature_params_update = []
            params_update = np.concatenate([pose_params_update, feature_params_update], axis=0)
            prev_params = self._all_params.copy()
            scale_angle_induced = 0.01 / max(0.01, np.abs(pose_params_update[2::3]).max())
            print('Angle induced scale:', scale_angle_induced)
            params_update *= scale_angle_induced
            print('L:', levenberg_iter, 'predicted chi2:', self._linearized_chi2(params_update))
            self._apply_update(params_update)
            self.compute_errors()
            chi2_after_update = self.get_chi2()
            print('L:', levenberg_iter, 'chi2:', self.get_chi2(), 'lambda:', self._lambda_value)
            rho = chi2_before - chi2_after_update
            rho_scale = (params_update * (self._lambda_value * params_update - b_combined)).sum() + 1e-5
            rho /= rho_scale
            if 0.0 < rho and np.isfinite(chi2_after_update):
                # Successfull update
                scale_factor = 1.0 - np.power((2.0 * rho - 1.0), 3)
                scale_factor = np.clip(scale_factor, 1.0 / 3.0, 2.0 / 3.0)
                self._lambda_value *= scale_factor
                self._lambda_factor = 2.0
                break
            # Unsuccessfull update. Roll back
            self._Hpp -= self._lambda_value * np.eye(*self._Hpp.shape)
            if 0 < len(self._features):
                self._Hll -= self._lambda_value * np.eye(*self._Hll.shape)
            self._all_params = prev_params
            self.compute_errors()
            self._lambda_value *= self._lambda_factor
            self._lambda_factor *= 2.0
            if 1E15 < self._lambda_value:
                self._lambda_value = None
                self._lambda_factor = 2.0
                break
            levenberg_iter += 1
        if levenberg_iter == LEVENBERG_ITERATIONS:
            print('L: Could not imporove chi2')


    @property
    def _all_vertices(self):
        return self._pose_vertices + [feature.vertex for feature in self._features]

    @property
    def _all_params(self):
        if len(self._all_vertices) == 0:
            return None
        return np.concatenate(list(map(lambda vertex: vertex.params, self._all_vertices)), axis=0)


    @_all_params.setter
    def _all_params(self, params):
        shift = 0
        for vertex in self._all_vertices:
            vertex.params = params[shift:(shift + vertex.dim)]
            shift += vertex.dim
        assert shift == len(params)


    def _apply_update(self, params_update):
        shift = 0
        for vertex in self._all_vertices:
            vertex.update(params_update[shift:(shift + vertex.dim)])
            shift += vertex.dim
        assert shift == len(params_update)


    def compute_errors(self):
        list(map(lambda edge: edge.compute_error(), self._all_edges))


    def get_chi2(self):
        return sum(map(lambda edge: edge.chi2(), self._all_edges))

    
    def _linearized_chi2(self, params_update):
        pose_var_dims = np.concatenate(
            [[0], np.cumsum([vertex.dim for vertex in self._pose_vertices])],
            axis=0)
        pose_vertex_to_index = dict(zip(self._pose_vertices, range(len(self._pose_vertices))))
        total_chi2 = 0.0
        for edge in self._pose_edges:
            vertex_indices = range(len(edge.vertices))
            local_to_global = lambda vertex_index: pose_vertex_to_index[edge.vertices[vertex_index]]
            var_range = lambda vertex_index: (pose_var_dims[local_to_global(vertex_index)],
                                              pose_var_dims[local_to_global(vertex_index) + 1])
            edge_error = edge.error
            for i in range(len(edge.vertices)):
                vertex_range = var_range(i)
                J = edge.J(i)
                edge_error += np.dot(edge.J(i), params_update[vertex_range[0]:vertex_range[1]])
            total_chi2 += np.dot(np.dot(edge_error, edge.inf), edge_error)
        if 0 < len(self._features):
            return total_chi2
        feature_var_dims = np.concatenate(
            [[0], np.cumsum([feature.vertex.dim for feature in self._features])],
            axis=0)
        feature_vertex_to_index = dict(
            zip([feature.vertex for feature in self._features], range(len(self._features))))
        for feature_index, feature in enumerate(self._features):
            feature_vertex_global_index = feature_vertex_to_index[feature.vertex]
            feature_var_range = (feature_var_dims[feature_vertex_global_index],
                                 feature_var_dims[feature_vertex_global_index + 1])
            for edge in feature.edges:
                assert edge.feature_vertex == feature.vertex
                pose_vertex_global_index = pose_vertex_to_index[edge.pose_vertex]
                pose_var_range = (pose_var_dims[pose_vertex_global_index],
                                  pose_var_dims[pose_vertex_global_index + 1])
                assert edge.vertices.index(edge.pose_vertex) == 0
                Jp = edge.J(0)
                Jf = edge.J(1)
                edge_error = edge.error
                edge_error += np.dot(Jp, params_update[pose_var_range[0]:pose_var_range[1]])
                edge_error += np.dot(Jf, params_update[feature_var_range[0]:feature_var_range[1]])
                total_chi2 += np.dot(np.dot(edge_error, edge.inf), edge_error)
        return total_chi2


    def optimize(self, steps):
        iteration = 0
        self.compute_errors()
        last_chi2 = self.get_chi2()
        print('Initial chi2:', last_chi2)
        while iteration < steps:
            self._linearize()
            self._assemble()
            self._solve_and_update()
            self.compute_errors()
            current_chi2 = self.get_chi2()
            print('Iteration', iteration, 'chi2:', current_chi2, 'delta:', last_chi2 - current_chi2)
            if 1E-3 <= last_chi2 - current_chi2:
                last_chi2 = current_chi2
                iteration += 1
                continue
            self._lambda_value = None
            return True
        self._lambda_value = None
        return False

   
    @property
    def features(self):
        return self._features


    @property
    def poses(self):
        assert 0 < len(self._pose_vertices), 'There are no poses'
        return np.array(list(map(operator.attrgetter('params'), self._pose_vertices)))
