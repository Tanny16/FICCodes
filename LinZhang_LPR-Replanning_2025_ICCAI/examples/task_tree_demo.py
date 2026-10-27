"""Offline task-tree subset extracted from the archived draft notebook."""
class TaskTreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_subtree(self, data):
        if self.data == data:
            return self

        for child in self.children:
            subtree = child.get_subtree(data)
            if subtree:
                return subtree
        return None

    def get_child_node(self, index):
        if 0 <= index < len(self.children):
            return self.children[index]
        else:
            return None

    def is_leaf(self):
        return len(self.children) == 0

    def are_all_leaves_int(self):
        if self.is_leaf():
            return isinstance(self.data, int)
        else:
            for child in self.children:
                if not child.are_all_leaves_int():
                    return False
            return True

    def find_non_int_leaves(self, path=[]):
        if self.is_leaf():
            if not isinstance(self.data, int):
                return [(self.data, path)]
        else:
            non_int_leaves = []
            for i, child in enumerate(self.children):
                child_path = path + [i]
                non_int_leaves.extend(child.find_non_int_leaves(child_path))
            return non_int_leaves
        return []

    def find_nodes_by_value(self, target_value, path=[]):
        nodes = []
        if self.data == target_value:
            nodes.append((self, path))
        for i, child in enumerate(self.children):
            child_path = path + [i]
            nodes.extend(child.find_nodes_by_value(target_value, child_path))
        return nodes

if __name__ == "__main__":
    root = TaskTreeNode("travel")
    root.add_child(TaskTreeNode(1))
    branch = TaskTreeNode("choose flight")
    branch.add_child(TaskTreeNode(2))
    root.add_child(branch)
    assert root.get_subtree("choose flight") is branch
    assert root.are_all_leaves_int()
    assert root.find_nodes_by_value(2)[0][1] == [1, 0]
    print("Task tree: subtree lookup, completed leaves and node paths passed.")
