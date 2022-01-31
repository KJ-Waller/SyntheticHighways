import sys
import xml.etree.ElementTree as ET
import clingo

class MapChanger(object):
    def __init__(self, xml_fname):
        self.xml_fname = xml_fname

    def suggest_changes(self):
        self.parse_xml()
        asp_program = self.generate_asp_prog()
        change_generator = self.run_asp(asp_program)
        roads_to_remove = next(change_generator)
        self.to_xml(roads_to_remove)
        ET.indent(self.xml_tree, space="\t", level=0)
        self.xml_tree.write("change_suggestions.xml", encoding="UTF-8", xml_declaration=True)
        print("change_suggestions.xml", end = '')
    
    def to_xml(self, roads_to_remove):
        root = ET.Element('root')
        remove_roads = ET.Element('RemoveRoads')
        root.append(remove_roads)

        for road in roads_to_remove:
            remove_road = ET.SubElement(remove_roads, 'RemoveRoad')
            remove_road.set('SegmentId', str(road[0]))
            remove_road.set('StartNodeId', str(road[1]))
            remove_road.set('EndNodeId', str(road[2]))
            remove_road.set('PrefabId', str(road[3]))
        
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
            start_nodeid = int(segment[0].attrib['NodeId'])
            end_nodeid = int(segment[1].attrib['NodeId'])
            if start_nodeid in node_ids and end_nodeid in node_ids:
                self.segments.append(segment)


    def generate_asp_prog(self):
        asp_program = ""
        for i, node in enumerate(self.nodes):
            asp_program += f"""node({node.attrib['Id']}).\n"""

        for i, segment in enumerate(self.segments):
            asp_program += f"edge({segment.attrib['SegmentId']},{segment[0].attrib['NodeId']},{segment[1].attrib['NodeId']}).\n"
        
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

    def run_asp(self, program):
        control = clingo.Control()
        control.add("base", [], program)
        control.ground([("base", [])])
    
        control.configuration.solve.models = 0
    
        prefab_ids = {int(segment.attrib['SegmentId']): int(segment.attrib['PrefabId']) for segment in self.segments}
        with control.solve(yield_=True) as handle:
            for model in handle:
                removed_roads = []
                for atom in model.symbols(shown=True):
                    if atom.name == 'removed_edge':
                        segment_id = int(atom.arguments[0].number)
                        prefab_id = prefab_ids[segment_id]
                        removed_roads.append((segment_id,int(atom.arguments[1].number), int(atom.arguments[2].number), prefab_id))
                yield removed_roads

if __name__ == "__main__":
    map_changer = MapChanger(sys.argv[1])
    map_changer.suggest_changes()
