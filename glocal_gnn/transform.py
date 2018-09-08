import graph_cpu


class TwoLocal(object):
    def __call__(self, data):
        out = graph_cpu.two_local(data.edge_index, data.x, data.num_nodes)
        data.edge_index_2, data.assignment_2, data.iso_type_2 = out
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ConnectedTwoLocal(object):
    def __call__(self, data):
        out = graph_cpu.connected_two_local(data.edge_index, data.x,
                                            data.num_nodes)
        data.edge_index_2, data.assignment_2, data.iso_type_2 = out
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class TwoMalkin(object):
    def __call__(self, data):
        out = graph_cpu.two_malkin(data.edge_index, data.x, data.num_nodes)
        data.edge_index_2, data.assignment_2, data.iso_type_2 = out
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ConnectedTwoMalkin(object):
    def __call__(self, data):
        out = graph_cpu.connected_two_malkin(data.edge_index, data.x,
                                             data.num_nodes)
        data.edge_index_2, data.assignment_2, data.iso_type_2 = out
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ThreeLocal(object):
    def __call__(self, data):
        out = graph_cpu.three_local(data.edge_index, data.x, data.num_nodes)
        data.edge_index_3, data.assignment_3, data.iso_type_3 = out
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ConnectedThreeLocal(object):
    def __call__(self, data):
        out = graph_cpu.connected_three_local(data.edge_index, data.x,
                                              data.num_nodes)
        data.edge_index_3, data.assignment_3, data.iso_type_3 = out
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ThreeMalkin(object):
    def __call__(self, data):
        out = graph_cpu.three_malkin(data.edge_index, data.x, data.num_nodes)
        data.edge_index_3, data.assignment_3, data.iso_type_3 = out
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class ConnectedThreeMalkin(object):
    def __call__(self, data):
        out = graph_cpu.connected_three_malkin(data.edge_index, data.x,
                                               data.num_nodes)
        data.edge_index_3, data.assignment_3, data.iso_type_3 = out
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class Assignment2To3(object):
    def __call__(self, data):
        out = graph_cpu.assignment_2to3(data.edge_index, data.num_nodes)
        data.assignment_2to3 = out
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
