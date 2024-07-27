class Node:
    def __init__(self, column=None, label=None, data=None, target=None):
        self.column = column
        self.label = label

        self.data = data
        self.target = target

        self.children = {}

    def __str__(self):
        return stringify_tree(self)

    def __repr__(self):
        return str(self)

def stringify_tree(node, indent="˪"):
        if not node.children: return f"{node.label}"
    
        out = []
        alter_pad = "  " * 2 
        for (k, v) in node.children.items():
            pad = ''
            if v.children:
                pad = alter_pad
                v = stringify_tree(v, indent = pad + indent + "-" * 4)

            out.append(
               f"{indent}{k}: \n{pad}{v}"
            )

        return "\n".join(out)