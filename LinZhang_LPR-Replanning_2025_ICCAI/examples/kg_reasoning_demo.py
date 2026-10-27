"""Offline subset extracted from RUN_code.ipynb cell 22; no API calls."""
import json
from pathlib import Path
import re

def read_KG_from_file(filename):
    KG = []
    with open(filename, 'r') as file:
        for line in file:
            # Remove the square brackets and split by comma
            triplet = [item.strip() for item in line.strip().replace("[", "").replace("]", "").split(',')]
            KG.append(triplet)
    return KG

# This function call will read your KG.txt and store each line as a list in KG

def find_paths(entity, kgs, current_path=None):
    if current_path is None:
        current_path = []

    # Find matches where the head of the KG matches the current entity
    #matches = [kg for kg in kgs if kg[0] == entity]
    pattern = re.compile(r'\b' + re.escape(entity.lower()) + r'\b')

    matches = [kg for kg in kgs if pattern.search(kg[0].lower())]


    # If no matches found, just return the current path as a list of paths
    if not matches:
        return [current_path]

    paths = []
    for match in matches:
        # Create a new path extending from the current path
        new_path = current_path + [match]
        paths.extend(find_paths(match[2], kgs, new_path))

    return paths

def build_tree(paths, limit = 0):
    # limit: only longest
    # Find the longest length among the paths
    #max_length = max(len(path) for path in paths)
    #print(max_length)
    # Filter out the longest path(s)
    #longest_paths = [path for path in paths if len(path) == max_length - limit]
    #pprint.pprint(longest_paths)
    #tree = defaultdict(dict)
    tree = {}

    def add_path_to_tree(node, path):
        if not path:
            return
        src, rel, dest = path[0]
        if src not in node:
            node[src] = {}
        if rel not in node[src]:
            node[src][rel] = {}
        if dest not in node[src][rel]:
            node[src][rel][dest] = {}
        add_path_to_tree(node[src][rel][dest], path[1:])

    for path in paths:
    #for path in longest_paths:
        add_path_to_tree(tree, path)

    return tree

def get_paths_dict(entities_list):
    paths_dict = {}

    for entity in entities_list:
        path = find_paths(entity, KG)

        path_tree = build_tree(paths = path)
        paths_dict[entity] = path_tree
    return paths_dict

if __name__ == "__main__":
    KG = read_KG_from_file(Path(__file__).resolve().parents[1] / "data/examples/KG.txt")
    paths = get_paths_dict(["My father", "Academic conference"])
    assert any(len(path) >= 2 for path in find_paths("My father", KG))
    print(json.dumps(paths, ensure_ascii=False, indent=2))
