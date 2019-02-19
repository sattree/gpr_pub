import nltk
import networkx as nx

def map_chars_to_tokens(doc, char_offset):
    #### Convert character level offsets to token level
    # tokenization or mention labelling may not be perfect
    # Identify the token that contains first character of mention
    # Identify the token that contains last character of mention
    # example - token: 'Delia-', mention: 'Delia'
    #           token: 'N.J.Parvathy', mention: 'Parvathy'
    # Token starts before the last character of mention and ends after the last character of mention
    # Remember character offset end here is the character immediately after the token
    return next(filter(lambda token: char_offset in range(token.idx, token.idx+len(token.text)), doc), None).i

def parse_tree_to_graph(sent_trees, doc, **kwargs):
    graph = nx.Graph() 
    leaves = []
    edges = []
    for sent_tree in sent_trees:
        edges, leaves = get_edges_in_tree(sent_tree, leaves=leaves, path='', edges=edges, **kwargs)
    graph.add_edges_from(edges)
    
    tokens = [token.word for token in doc]
    assert tokens == leaves, 'Tokens in parse tree and input sentence don\'t match.'
    
    return graph
    
#DFS
# trace path to create unique names for all nodes
def get_edges_in_tree(parent, leaves=[], path='', edges=[], lrb_rrb_fix=False):
    for i, node in enumerate(parent):
        if type(node) is nltk.Tree:
            from_node = path
            to_node = '{}-{}-{}'.format(path, node.label(), i)
            edges.append((from_node, to_node))

            if lrb_rrb_fix:
	            if node.label() == '-LRB-':
	                leaves.append('(')
	            if node.label() == '-RRB-':
	                leaves.append(')')

            edges, leaves = get_edges_in_tree(node, leaves, to_node, edges)
        else:
            from_node = path
            to_node = '{}-{}'.format(node, len(leaves))
            edges.append((from_node, to_node))
            leaves.append(node)
    return edges, leaves

def get_syntactical_distance_from_graph(graph, token_a, token_b, debug=False):
       return nx.shortest_path_length(graph, 
                                   source='{}-{}'.format(token_a.word, token_a.i),
                                   target='{}-{}'.format(token_b.word, token_b.i))

def get_normalized_tag(token):
        tag = token.dep_
        tag = 'subj' if 'subj' in tag else tag
        tag = 'dobj' if 'dobj' in tag else tag
        return tag