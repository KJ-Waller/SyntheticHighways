import sys
import xml.etree.ElementTree as ET
import clingo

PREFAB_DICT = {
    35: {
        'name': 'Basic Road',
        'lanes': 1,
    },
    37: {
        'name': 'Large Road',
        'lanes': 3,
    },
    38: {
        'name': 'Gravel Road',
        'lanes': 1
    },
    40: {
        'name': 'Medium Road',
        'lanes': 2,
    },
}

class MapChanger(object):
    def __init__(self, xml_fname):
        self.xml_fname = xml_fname
        self.remove_roads = False

    def suggest_changes(self):
        self.parse_xml()
        if self.remove_roads:
            asp_program = self.generate_asp_prog_v1()
        else:
            asp_program = self.generate_asp_prog_v2()
        change_generator = self.run_asp(asp_program)
        changes = next(change_generator)
        self.to_xml(changes)
        ET.indent(self.xml_tree, space="\t", level=0)
        map_name = self.xml_fname[self.xml_fname.find('/')+1:self.xml_fname.rfind('_temp')]
        changes_fname = f"./SyntheticHighways/{map_name}_changes.xml"
        self.xml_tree.write(changes_fname, encoding="UTF-8", xml_declaration=True)
        print(changes_fname, end = '')
    
    def to_xml(self, road_changes):
        initial_changes = road_changes['initial_changes']
        snapshot_changes = road_changes['changes']

        root = ET.Element('root')
        
        # Write initial changes to XML
        init_changes = ET.Element('InitialChanges')
        root.append(init_changes)

        for added_road in initial_changes['additions']:
            add_road = ET.SubElement(init_changes, 'AddRoad')
            add_road.set('SegmentId', str(added_road[0]))
            add_road.set('StartNodeId', str(added_road[1]))
            add_road.set('EndNodeId', str(added_road[2]))
            add_road.set('PrefabId', str(added_road[3]))

        for removed_road in initial_changes['removals']:
            remove_road = ET.SubElement(init_changes, 'RemoveRoad')
            remove_road.set('SegmentId', str(removed_road[0]))
            remove_road.set('StartNodeId', str(removed_road[1]))
            remove_road.set('EndNodeId', str(removed_road[2]))
            remove_road.set('PrefabId', str(removed_road[3]))

        for road_change in initial_changes['prefab_changes']:
            road_chng = ET.SubElement(init_changes, 'PrefabChange')
            road_chng.set('SegmentId', str(road_change[0]))
            road_chng.set('StartNodeId', str(road_change[1]))
            road_chng.set('EndNodeId', str(road_change[2]))
            road_chng.set('OldPrefabId', str(road_change[3]))
            road_chng.set('NewPrefabId', str(road_change[4]))

        # Write snapshot changes to XML
        
        snpsht_changes = ET.Element('SnapshotChanges')
        root.append(snpsht_changes)

        for added_road in snapshot_changes['additions']:
            add_road = ET.SubElement(snpsht_changes, 'AddRoad')
            add_road.set('SegmentId', str(added_road[0]))
            add_road.set('StartNodeId', str(added_road[1]))
            add_road.set('EndNodeId', str(added_road[2]))
            add_road.set('PrefabId', str(added_road[3]))

        for removed_road in snapshot_changes['removals']:
            remove_road = ET.SubElement(snpsht_changes, 'RemoveRoad')
            remove_road.set('SegmentId', str(removed_road[0]))
            remove_road.set('StartNodeId', str(removed_road[1]))
            remove_road.set('EndNodeId', str(removed_road[2]))
            remove_road.set('PrefabId', str(removed_road[3]))

        for road_change in snapshot_changes['prefab_changes']:
            road_chng = ET.SubElement(snpsht_changes, 'PrefabChange')
            road_chng.set('SegmentId', str(road_change[0]))
            road_chng.set('StartNodeId', str(road_change[1]))
            road_chng.set('EndNodeId', str(road_change[2]))
            road_chng.set('OldPrefabId', str(road_change[3]))
            road_chng.set('NewPrefabId', str(road_change[4]))
        
        self.xml_tree = ET.ElementTree(root)

    def parse_xml(self, service='Road'):
        map_tree = ET.parse(self.xml_fname)
        map_root = map_tree.getroot()
        nodes = map_root[0]
        segments = map_root[1]

        node_ids = []
        self.nodes = []

        for node in nodes:
            if node.attrib['Service'] == service:
                self.nodes.append(node)
                node_ids.append(int(node.attrib['Id']))

        self.segments = []
        for segment in segments:
            if int(segment.attrib['PrefabId']) not in PREFAB_DICT.keys():
                continue
            start_nodeid = int(segment[0].attrib['NodeId'])
            end_nodeid = int(segment[1].attrib['NodeId'])
            if start_nodeid in node_ids and end_nodeid in node_ids:
                self.segments.append(segment)


    def generate_asp_prog_v1(self):
        asp_program = ""
        for i, node in enumerate(self.nodes):
            asp_program += f"""node({node.attrib['Id']}).\n"""

        for i, segment in enumerate(self.segments):
            asp_program += f"edge({segment.attrib['SegmentId']},{segment[0].attrib['NodeId']},{segment[1].attrib['NodeId']}).\n"
            asp_program += f"edge_type({segment.attrib['SegmentId']},{segment.attrib['PrefabId']}).\n"
        
        asp_program += f"""
        % Every edge can be a new edge
        {{ new_edge(EN,N1,N2) }} 1 :- edge(EN,N1,N2).
    
        % Maximize the number of roads to be removed
        numedge_old(OC) :- OC = #count {{ EN,edge(EN,N1,N2) : edge(EN,N1,N2) }}.
        numedge_new(NC) :- NC = #count {{ EN,new_edge(EN,N1,N2) : new_edge(EN,N1,N2) }}.
        #maximize {{ DIFF : numedge_old(OC), numedge_new(NC), DIFF = OC - NC}}.
    
        % Define reachability over old network
        reachable(N1,N1) :- node(N1).
        reachable(N1,N2) :- node(N1), node(N2), edge(_,N1,N2).
        reachable(N1,N3) :- node(N1), node(N2), node(N3), N1!=N2, N2!=N3, reachable(N1,N2), reachable(N2,N3).
    
        % Define reachability over new network
        new_reachable(N1,N1) :- node(N1).
        new_reachable(N1,N2) :- node(N1), node(N2), new_edge(_,N1,N2).
        new_reachable(N1,N3) :- node(N1), node(N2), node(N3), N1!=N2, N2!=N3, new_reachable(N1,N2), new_reachable(N2,N3).
    
        % Define node degrees
        %degree(N1,D) :- node(N1), D = OD + ID,
        %  OD = #count {{ EN : node(N1), edge(EN,N1,N2) }}, 
        %  ID = #count {{ EN : node(N1), edge(EN,N2,N1) }}.
        % degree(N1,D) :- node(N1), D = #count {{ EN : node(N1), edge(EN,N2,N1) }}.
    
        :- reachable(N1,N2), not new_reachable(N1,N2).
        %:- reachable(N1,N2), not new_reachable(N1,N2), degree(N1,D1), degree(N2,D2), D1 > 2, D2 > 2.
    
        removed_edge(EN,N1,N2) :- edge(EN,N1,N2), not new_edge(EN,N1,N2).
        """

        return asp_program

        
    def generate_asp_prog_v2(self, prefab_change_fraction=0.1):
        asp_program = ""
        
        num_prefab_changes = int(len(self.segments) * prefab_change_fraction)
        asp_program += f"#const n={num_prefab_changes}.\n"

        prefab_ids = PREFAB_DICT.keys()
        for pid in prefab_ids:
            asp_program += f"pid({pid})."

        for i, node in enumerate(self.nodes):
            asp_program += f"""node({node.attrib['Id']}).\n"""

        for i, segment in enumerate(self.segments):
            asp_program += f"edge({segment.attrib['SegmentId']},{segment[0].attrib['NodeId']},{segment[1].attrib['NodeId']}).\n"
            asp_program += f"edge_type({segment.attrib['SegmentId']},{segment.attrib['PrefabId']}).\n"

        
        asp_program += f"""
        % Every edge can be a new edge
        {{ new_edge(EN,N1,N2) }} 1 :- edge(EN,N1,N2).

        % Maximize the number of roads to be removed
        numedge_old(OC) :- OC = #count {{ EN,edge(EN,N1,N2) : edge(EN,N1,N2) }}.
        numedge_new(NC) :- NC = #count {{ EN,new_edge(EN,N1,N2) : new_edge(EN,N1,N2) }}.
        #maximize {{ DIFF : numedge_old(OC), numedge_new(NC), DIFF = OC - NC}}.

        % Define reachability over old network
        reachable(N1,N1) :- node(N1).
        reachable(N1,N2) :- node(N1), node(N2), edge(_,N1,N2).
        reachable(N1,N3) :- node(N1), node(N2), node(N3), N1!=N2, N2!=N3, reachable(N1,N2), reachable(N2,N3).

        % Define reachability over new network
        new_reachable(N1,N1) :- node(N1).
        new_reachable(N1,N2) :- node(N1), node(N2), new_edge(_,N1,N2).
        new_reachable(N1,N3) :- node(N1), node(N2), node(N3), N1!=N2, N2!=N3, new_reachable(N1,N2), new_reachable(N2,N3).

        % Define node degrees
        %degree(N1,D) :- node(N1), D = OD + ID,
        %  OD = #count {{ EN : node(N1), edge(EN,N1,N2) }}, 
        %  ID = #count {{ EN : node(N1), edge(EN,N2,N1) }}.
        % degree(N1,D) :- node(N1), D = #count {{ EN : node(N1), edge(EN,N2,N1) }}.

        :- reachable(N1,N2), not new_reachable(N1,N2).
        %:- reachable(N1,N2), not new_reachable(N1,N2), degree(N1,D1), degree(N2,D2), D1 > 2, D2 > 2.

        added_edge(EN,N1,N2) :- edge(EN,N1,N2), not new_edge(EN,N1,N2).

        1 {{ new_edge_type(SID, PID) : pid(PID) }} 1 :- new_edge(SID,_,_).
        num_road_changes(N) :- N = #count {{ SID1 : new_edge_type(SID1, PID1), edge_type(SID2, PID2), PID1 != PID2, SID1 == SID2}}.

        :- num_road_changes(N), n != N.

        road_type_changes(SID, N1, N2, PID1, PID2) :- edge_type(SID, PID1), new_edge_type(SID, PID2), PID1 != PID2, new_edge(SID,N1,N2).

        removed_edge(EN,N1,N2) :- edge(EN,N1,N2), not new_edge(EN,N1,N2).
        """
        
        return asp_program

    def run_asp(self, program):
        control = clingo.Control()
        control.add("base", [], program)
        control.ground([("base", [])])
    
        control.configuration.solve.models = 0
    
        prefab_ids = {int(segment.attrib['SegmentId']): int(segment.attrib['PrefabId']) for segment in self.segments}
        with control.solve(yield_=True) as handle:
            for model in handle:
                removed_roads = []
                prefab_changes = []
                for atom in model.symbols(shown=True):
                    if atom.name == 'removed_edge':
                        segment_id = int(atom.arguments[0].number)
                        prefab_id = prefab_ids[segment_id]
                        removed_roads.append((segment_id,int(atom.arguments[1].number), int(atom.arguments[2].number), prefab_id))
                    if atom.name == 'road_type_changes':
                        segment_id = int(atom.arguments[0].number)
                        n1 = int(atom.arguments[1].number)
                        n2 = int(atom.arguments[2].number)
                        old_prefab_id = int(atom.arguments[3].number)
                        new_prefab_id = int(atom.arguments[4].number)
                        prefab_changes.append((segment_id, n1, n2, old_prefab_id, new_prefab_id))
                yield {
                    'initial_changes': {
                        'additions': [],
                        'removals': [],
                        'prefab_changes': []
                    },
                    'changes': {
                        'additions': [],
                        'removals': removed_roads,
                        'prefab_changes': prefab_changes
                    }
                }

if __name__ == "__main__":
    map_changer = MapChanger(sys.argv[1])
    map_changer.suggest_changes()
