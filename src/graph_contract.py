"""Canonical graph boundary for HGWM scoring. No model or simulator imports."""
from __future__ import annotations

import copy
import re
from collections import defaultdict

SCHEMA_VERSION = 1
UNKNOWN = {'', 'unknown', 'none', 'null'}

def canonical_name(value):
    if value is None:
        return ''
    return re.sub(r'[\s-]+', '_', str(value).strip().lower())

def _records(value):
    if isinstance(value, dict):
        return [dict(v, id=v.get('id', k), name=v.get('name', k))
                if isinstance(v, dict) else {'id': k, 'name': str(v)}
                for k, v in value.items()]
    if isinstance(value, (list, tuple)):
        return [copy.deepcopy(v) if isinstance(v, dict) else {'name': str(v)} for v in value]
    return []

def _relation(value):
    name = canonical_name(value)
    return {'next_to': 'near', 'beside': 'near', 'connected_to': 'connected',
            'contains_object': 'contains'}.get(name, name or 'related')

def normalize_scene_graph(graph):
    """Expose flat and typed views of the same nodes and edges.

    Object IDs and attributes are preserved; room IDs use their normalized labels.
    Rooms/containment are derived only
    from observed room membership. Unresolvable edges are diagnosed, never guessed.
    A normalized graph's flat edge list is authoritative after relation edits.
    """
    if not isinstance(graph, dict):
        raise TypeError('scene graph must be a dictionary')
    result = copy.deepcopy(graph)
    room_records = _records(graph.get('room_nodes', {}))
    object_records = _records(graph.get('object_nodes', []))
    flat = _records(graph.get('nodes', []))
    records = [(v, 'room') for v in room_records] + [(v, 'object') for v in object_records]
    records += [(v, 'room' if canonical_name(v.get('type')) == 'room' else 'object') for v in flat]
    objects, rooms, id_aliases = {}, {}, {}
    for index, (raw, kind) in enumerate(records):
        node = copy.deepcopy(raw)
        if kind == 'room':
            name = canonical_name(node.get('room_type') or node.get('name') or node.get('id'))
            if name in UNKNOWN:
                continue
            original_id = str(node.get('id') or 'room:' + name)
            node_id = 'room:' + name
            id_aliases[original_id] = node_id
            node.update(id=node_id, name=name, type='room', room_type=name)
            node.setdefault('avg_confidence', node.get('confidence', 0.5))
            rooms[name] = node
        else:
            name = canonical_name(node.get('name') or node.get('id'))
            if not name:
                continue
            node_id = str(node.get('id') or f'object:{name}:{index}')
            room = canonical_name(node.get('room') or node.get('room_type') or node.get('parent_room') or 'unknown')
            node.update(id=node_id, name=name, type='object', room=room, room_type=room, parent_room=room)
            if node_id in objects:
                objects[node_id].update(node)
            else:
                objects[node_id] = node
    for obj in objects.values():
        room = obj['room']
        if room not in UNKNOWN and room not in rooms:
            rooms[room] = {'id': 'room:' + room, 'name': room, 'type': 'room', 'room_type': room,
                           'avg_confidence': obj.get('confidence', 0.5)}
    nodes = list(rooms.values()) + list(objects.values())
    by_id = {n['id']: n for n in nodes}
    by_name = defaultdict(list)
    for node in nodes:
        by_name[node['name']].append(node['id'])
    def resolve(value):
        value = str(value)
        value = id_aliases.get(value, value)
        if value in by_id:
            return value
        choices = by_name.get(canonical_name(value), [])
        return choices[0] if len(choices) == 1 else None
    raw_edges = _records(graph.get('edges', []))
    if graph.get('_graph_schema_version') != SCHEMA_VERSION:
        for key in ['room_edges', 'object_edges', 'room_object_edges']:
            raw_edges += _records(graph.get(key, []))
    edges = {}
    rejected = copy.deepcopy(graph.get('schema_diagnostics', {}).get('rejected_edges', [])) if graph.get('_graph_schema_version') == SCHEMA_VERSION else []
    for raw in raw_edges:
        source = resolve(raw.get('source', raw.get('from', '')))
        target = resolve(raw.get('target', raw.get('to', '')))
        if source is None or target is None:
            rejected.append({'source': str(raw.get('source', raw.get('from', ''))),
                             'target': str(raw.get('target', raw.get('to', ''))),
                             'reason': 'missing or ambiguous endpoint'})
            continue
        relation = _relation(raw.get('relation', raw.get('type', 'related')))
        edge = dict(raw, source=source, target=target, relation=relation, type=relation)
        edge['from'], edge['to'] = source, target
        edge.setdefault('confidence', raw.get('weight', 0.5))
        edges[(source, relation, target)] = edge
    for obj in objects.values():
        if obj['room'] in rooms:
            source, target = rooms[obj['room']]['id'], obj['id']
            key = (source, 'contains', target)
            edges.setdefault(key, {'source': source, 'target': target, 'from': source, 'to': target,
                                   'relation': 'contains', 'type': 'contains',
                                   'confidence': obj.get('confidence', 0.5)})
    room_edges, object_edges, cross_edges = [], [], []
    for edge in edges.values():
        kinds = (by_id[edge['source']]['type'], by_id[edge['target']]['type'])
        if kinds == ('room', 'room'):
            room_edges.append(edge)
        elif kinds == ('object', 'object'):
            object_edges.append(edge)
        else:
            cross_edges.append(edge)
    depth_layers = {'foreground': [], 'mid-ground': [], 'background': []}
    for obj in objects.values():
        category = obj.get('depth_info', {}).get('depth_category')
        if category in depth_layers:
            depth_layers[category].append(obj)
    result.update(nodes=nodes, edges=list(edges.values()), room_nodes=rooms,
                  object_nodes=list(objects.values()), room_edges=room_edges,
                  object_edges=object_edges, room_object_edges=cross_edges,
                  spatial_relationships=list(edges.values()), depth_layers=depth_layers,
                  _graph_schema_version=SCHEMA_VERSION,
                  schema_diagnostics={'room_count': len(rooms), 'object_count': len(objects),
                                      'edge_count': len(edges), 'rejected_edges': rejected})
    return result

def merge_scene_graphs(memory, current=None):
    """Merge by stable ID; current attributes take precedence, never by name alone."""
    left = normalize_scene_graph(memory)
    if current is None:
        return left
    right = normalize_scene_graph(current)
    nodes = {n['id']: copy.deepcopy(n) for n in left['nodes']}
    for n in right['nodes']:
        nodes.setdefault(n['id'], {}).update(n)
    return normalize_scene_graph({'nodes': list(nodes.values()), 'edges': left['edges'] + right['edges']})

def normalize_goal_graph(graph, goal=''):
    """Bridge the legacy room/anchor prompt schema and explicit graph records."""
    if not isinstance(graph, dict):
        raise TypeError('goal graph must be a dictionary')
    result = copy.deepcopy(graph)
    hierarchy = result.get('target_room_hierarchy') or result.get('target_room_connection') or {}
    result['target_room_hierarchy'] = hierarchy
    nodes = _records(result.get('nodes', []))
    edges = _records(result.get('edges', []))
    room_names = []
    if isinstance(hierarchy, dict):
        room_names += [hierarchy.get('primary_room')]
        for key in ['secondary_rooms', 'adjacent_zones']:
            value = hierarchy.get(key, [])
            room_names += [value] if isinstance(value, str) else list(value or [])
    connections = result.get('hallway_navigation', {}).get('room_connections', {})
    for connection in _records(connections):
        source = canonical_name(connection.get('from', connection.get('source', '')))
        target = canonical_name(connection.get('to', connection.get('target', '')))
        if source and target:
            room_names += [source, target]
            edges.append({'source': 'room:' + source, 'target': 'room:' + target,
                          'relation': connection.get('relation', 'connected')})
    known_rooms = {canonical_name(n) for n in room_names} - UNKNOWN
    for node in nodes:
        if canonical_name(node.get('name')) in known_rooms:
            node['type'] = 'room'
            node['room_type'] = canonical_name(node['name'])
    for name in sorted(known_rooms):
        nodes.append({'id': 'room:' + name, 'name': name, 'type': 'room', 'room_type': name})
    anchors = result.get('spatial_anchor_objects', [])
    if isinstance(anchors, dict):
        anchors = [dict(v, name=v.get('name', k)) if isinstance(v, dict) else {'name': k}
                   for k, v in anchors.items()]
    for anchor in _records(anchors):
        name = canonical_name(anchor.get('name'))
        if name:
            anchor.update(id=anchor.get('id', 'goal:' + name), name=name, type='object')
            nodes.append(anchor)
    target = canonical_name(goal or result.get('target_object', ''))
    if target:
        nodes.append({'id': 'goal:' + target, 'name': target, 'type': 'object'})
    normalized = normalize_scene_graph(dict(nodes=nodes, edges=edges))
    result.update(nodes=normalized['nodes'], edges=normalized['edges'],
                  target_object=target, _graph_schema_version=SCHEMA_VERSION)
    return result

def relation_keys(graph, kind=None):
    graph = normalize_scene_graph(graph)
    by_id = {n['id']: n for n in graph['nodes']}
    edges = graph['edges'] if kind is None else graph.get(kind, [])
    keys = set()
    for edge in edges:
        a, b = by_id[edge['source']]['name'], by_id[edge['target']]['name']
        relation = edge['relation']
        if relation in {'near', 'adjacent', 'connected'}:
            a, b = sorted([a, b])
        keys.add((a, relation, b))
    return keys

def relation_match_score(scene, goal, kind):
    observed, expected = relation_keys(scene, kind), relation_keys(goal, kind)
    if not expected or not observed:
        return 0.0
    return len(observed & expected) / max(len(observed), len(expected))
