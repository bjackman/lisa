import unittest
from collections import OrderedDict

from hypothesis import given
import hypothesis.strategies as st

from energy_model import (_CpuTree, EnergyModel,
                          ActiveState, EnergyModelNode, PowerDomain)


# Generate a nested list of Nones
nested_list_none = st.recursive(
    st.none(), lambda children: st.lists(children, min_size=1, max_size=8)
    ).filter(lambda l: l is not None)

@st.composite
def cpu_tree(draw, elements=nested_list_none):
    l = draw(elements)

    def recurse(l, leaf_count, node_count):
        new_l = []
        for idx, val in enumerate(l):
            if val is None:
                new_l.append(_CpuTree(cpu=leaf_count, children=None))
                leaf_count += 1
            else:
                sub_l, leaf_count, node_count = recurse(val,
                                                        leaf_count, node_count)
                new_l.append(_CpuTree(children=sub_l, cpu=None))
            node_count += 1
        return new_l, leaf_count, node_count

    children, leaf_count, node_count = recurse(l, 0, 0)
    tree = _CpuTree(children=children, cpu=None)
    return tree, leaf_count, node_count + 1

class TestCpuTree(unittest.TestCase):
    @given(cpu_tree())
    def test_cpu_tree_iter_leaves(self, data):
        tree, leaf_count, node_count = data

        leaves = list(tree.iter_leaves())

        self.assertEqual(len(leaves), leaf_count)

        for n in leaves:
            self.assertEqual(len(n.cpus), 1)

        cpus = [leaf.cpus[0] for leaf in leaves]
        self.assertEqual(cpus, range(leaf_count))

    @given(cpu_tree())
    def test_cpu_tree_iter_nodes(self, data):
        tree, leaf_count, node_count = data

        nodes = list(tree.iter_nodes())

        self.assertEqual(len(nodes), node_count)

# nosetests doesn't run this for some reason
if __name__ == "__main__":
    unittest.main()
