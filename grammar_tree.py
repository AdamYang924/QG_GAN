from collections import defaultdict

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.height = None

    def add_child(self, child_node):
        self.children.append(child_node)

    def remove_child(self, child_node):
        self.children = [child for child in self.children if child is not child_node]
    
    def calculate_height(self):
        if self.height is not None:
            return self.height
        if not self.children:
            self.height = 0  # Leaf node, height is 0
        else:
            # Height is 1 + the maximum height of any child
            self.height = 1 + max(child.calculate_height() for child in self.children)
        return self.height

    def __repr__(self, level=0):
        ret = "  " * level + repr(self.data) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret

class GTree:
    def __init__(self, root_data):
        self.root = TreeNode(root_data)
        self.parent_dic = defaultdict(list)
        self.node_dic = defaultdict(list)
        self.node_dic[root_data].append(self.root)

    def __repr__(self):
        return repr(self.root)

    def find(self, data):
        return self._find_recursively(data, self.root)

    # def find(self,data):
    #     result = self.node_dic[data]
    #     # assert len(result)<= 1, "Expect result <= 1"
    #     return result[0] if len(result)>0 else None
    
    def _find_recursively(self, data, node):
        if node.data == data:
            return node
        for child in node.children:
            result = self._find_recursively(data, child)
            if result:
                return result
        return None

    def add(self, parent_data, child_data):
        parent_nodes = self.node_dic[parent_data]
        if len(parent_nodes) > 0:
            node = TreeNode(child_data)
            self.node_dic[child_data].append(node)
            for parent_node in parent_nodes:
                parent_node.add_child(node)
                self.parent_dic[child_data].append(parent_node)
        else:
            raise ValueError(f"Parent node with data '{parent_data}' not found.")

    # def remove(self, data):
    #     if self.root.data == data:
    #         self.root = None
    #     else:
    #         parent_node, node_to_remove = self.find_parent(data)
    #         if node_to_remove:
    #             parent_node.remove_child(node_to_remove)
    #             self.parent_dic[data].remove(parent_node)
    #         else:
    #             raise ValueError(f"Node with data '{data}' not found.")

    # def find_parent(self, data, node, parent=None):
    #     if node.data == data:
    #         return parent, node
    #     for child in node.children:
    #         result = self.find_parent(data, child, node)
    #         if result[1]:
    #             return result
    #     return None, None
    def find_parent(self,data):
        result = self.parent_dic[data]
        result = sorted(result,key=TreeNode.calculate_height,reverse=True)
        return result[0]

    def traverse(self, node=None):
        if node is None:
            node = self.root
        print(node.data)
        for child in node.children:
            self.traverse(child)

if __name__ == "__main__":
    # Create a tree with a root node
    tree = GTree("Root")

    # Add children to the root node
    tree.add("Root", "Child 1")
    tree.add("Root", "Child 2")

    # Add children to "Child 1"
    tree.add("Child 1", "Grandchild 1.1")
    tree.add("Child 1", "Grandchild 1.2")

    # Add children to "Child 2"
    tree.add("Child 2", "Grandchild 2.1")

    # Print the tree structure
    print(tree)

    # Traverse the tree
    print("Tree traversal:")
    tree.traverse()

    # Find a node
    node = tree.find("Grandchild 1.1")
    print(f"Found node: {node.data}")

    # Remove a node
    tree.remove("Child 2")
    print("Tree after removing 'Child 2':")
    print(tree)
