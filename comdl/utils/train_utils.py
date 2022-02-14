

def update_cfg(tree,key,value, if_present = ""):
    """Updates the entries for a nested dictionary"""
    if key in tree and (if_present == "" or if_present in tree) and tree[key] != value:
        tree[key] = value
        return True
    for branch in tree.values():
        if isinstance(branch, dict):
            if update_cfg(branch,key,value, if_present):
                return True
        if isinstance(branch, list): #sometimes lists of dicts
            for it in branch:
                if isinstance(it, dict): 
                    if update_cfg(it,key,value, if_present):
                        return True
    return False

def add_key(tree, in_key, new_key, value):
    if in_key in tree:
        tree[new_key] = value
        return True
    for branch in tree.values():
        if isinstance(branch, dict):
            if add_key(branch,in_key,new_key, value):
                return True
        if isinstance(branch, list): #sometimes lists of dicts
            for it in branch:
                if isinstance(it, dict): 
                    if add_key(branch,in_key,new_key, value):
                        return True

def is_key_present(tree,key):
    if key in tree:
        return True
    for branch in tree.values():
        if isinstance(branch, dict):
            if is_key_present(branch,key):
                return True
        if isinstance(branch, list): #sometimes lists of dicts
            for it in branch:
                if isinstance(it, dict): 
                    if is_key_present(branch,key):
                        return True
    return False

def change_key(dc, key, val, if_present = ""):
    while True:
        res = update_cfg(dc, key, val, if_present)
        if not res:
            break

def change_keys(cfg, x:list) -> None:
        for it in x:
            change_key(cfg, it[0], it[1], it[2])