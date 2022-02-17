using System.Collections.Generic;
using System.Xml;
using UnityEngine;
using ColossalFramework.Plugins;
using ColossalFramework;
using System.IO;

namespace SyntheticHighways.MapExporter
{
    class MapExporter : MonoBehaviour
    {
        NetManager nManager;
        PathManager pManager;
        XmlDocument mapDoc;
        private Dictionary<int, XmlElement> nodeDictionary;

        void Start()
        {
            // Initialize network and path managers
            nManager = Singleton<NetManager>.instance;
            pManager = Singleton<PathManager>.instance;
        }

        public XmlDocument ExportMap(int snapNumber, bool saveXML = true)
        {
            // Initialize node dictionary for keeping track of nodes and their keys
            nodeDictionary = new Dictionary<int, XmlElement>();

            // Initialize XML file for map snapshot
            mapDoc = new XmlDocument();
            XmlDeclaration xmlDeclaration = mapDoc.CreateXmlDeclaration("1.0", "UTF-8", null);
            mapDoc.AppendChild(xmlDeclaration);
            XmlElement root = mapDoc.CreateElement("Map");
            mapDoc.AppendChild(root);

            // Export nodes and segments
            ExportNodes();
            ExportSegments();

            // Save results to XML file
            if (saveXML)
            {
                string cityName = (string)((SimulationMetaData)Singleton<SimulationManager>.instance.m_metaData).m_CityName;
                string currDir = Directory.GetCurrentDirectory();
                string folder = Path.Combine(currDir, "SyntheticHighways");

                if (!Directory.Exists(folder))
                {
                    Directory.CreateDirectory(folder);
                }

                // Save trajectories to XML file
                string fname = cityName + "_" + snapNumber.ToString() + "_map.xml";
                string saveName = Path.Combine(folder, fname);

                mapDoc.Save(saveName);
                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Map snapshot saved to: " + saveName);
            }

            return mapDoc;
        }

        void ExportNodes()
        {
            Singleton<NetManager>.Ensure();
            XmlElement nodes = mapDoc.CreateElement("Nodes");
            int key = 0;
            foreach (NetNode netNode in (NetNode[])((Array16<NetNode>)nManager.m_nodes).m_buffer)
            {
                if (EnumExtensions.IsFlagSet<NetNode.Flags>((NetNode.Flags)netNode.m_flags, (NetNode.Flags)1))
                {
                    // Initialize node
                    XmlElement node = mapDoc.CreateElement("Node");
                    
                    // Add ID attribute
                    XmlAttribute Id = mapDoc.CreateAttribute("Id");
                    Id.Value = key.ToString();
                    node.Attributes.Append(Id);

                    // Add location element with x,y,z attributes to node element
                    XmlElement loc = mapDoc.CreateElement("Location");
                    XmlAttribute x = mapDoc.CreateAttribute("x");
                    x.Value = netNode.m_position.x.ToString();
                    loc.Attributes.Append(x);
                    XmlAttribute y = mapDoc.CreateAttribute("y");
                    y.Value = netNode.m_position.y.ToString();
                    loc.Attributes.Append(y);
                    XmlAttribute z = mapDoc.CreateAttribute("z");
                    z.Value = netNode.m_position.z.ToString();
                    loc.Attributes.Append(z);
                    node.AppendChild(loc);

                    // Get node info
                    NetInfo info = netNode.Info;

                    // Add road type as service
                    XmlAttribute service = mapDoc.CreateAttribute("Service");
                    service.Value = ((PrefabInfo)info).GetService().ToString();
                    node.Attributes.Append(service);

                    // Add more explicit roat type as subservice
                    XmlAttribute subservice = mapDoc.CreateAttribute("SubService");
                    subservice.Value = ((PrefabInfo)info).GetSubService().ToString();
                    node.Attributes.Append(subservice);

                    // Add elevation
                    XmlAttribute elevation = mapDoc.CreateAttribute("Elevation");
                    elevation.Value = netNode.m_elevation.ToString();
                    node.Attributes.Append(elevation);

                    // TODO: Add if underground or above ground
                    /*xmlNode.UnderGround = EnumExtensions.IsFlagSet<NetNode.Flags>((NetNode.Flags)netNode.m_flags, (NetNode.Flags)524288);
                    xmlNode.OnGround = EnumExtensions.IsFlagSet<NetNode.Flags>((NetNode.Flags)netNode.m_flags, (NetNode.Flags)16384);*/

                    // Append node to nodes
                    nodes.AppendChild(node);

                    // Add node and key to dictionary for when checking segments in ExportSegments
                    nodeDictionary.Add(key, node);
                }
                ++key;
            }
            XmlElement root = mapDoc.DocumentElement;
            root.AppendChild(nodes);
        }

        void ExportSegments()
        {
            XmlElement segments = mapDoc.CreateElement("Segments");

            int key = 0;
            pManager.WaitForAllPaths();
            foreach (NetSegment netSegment in (NetSegment[])((Array16<NetSegment>)nManager.m_segments).m_buffer)
            {
                if (EnumExtensions.IsFlagSet<NetSegment.Flags>((NetSegment.Flags)netSegment.m_flags, (NetSegment.Flags)1))
                {
                    // Create segment element, adding 2 children: the starting node location, and end node location
                    // First create the segment element, assigning it a unique ID, adding a prefabid and number of lanes
                    XmlElement segment = mapDoc.CreateElement("Segment");
                    XmlAttribute segmentId = mapDoc.CreateAttribute("SegmentId");
                    segmentId.Value = key.ToString();
                    segment.Attributes.Append(segmentId);

                    XmlAttribute prefabId = mapDoc.CreateAttribute("PrefabId");
                    prefabId.Value = netSegment.Info.m_prefabDataIndex.ToString();
                    segment.Attributes.Append(prefabId);

                    NetInfo ni = PrefabCollection<NetInfo>.GetPrefab((uint)netSegment.Info.m_prefabDataIndex);
                    XmlAttribute prefabNameAttrib = mapDoc.CreateAttribute("PrefabName");
                    prefabNameAttrib.Value = ni.name;
                    segment.Attributes.Append(prefabNameAttrib);

                    int forwardLanes = 0;
                    int backwardLanes = 0;
                    netSegment.CountLanes((ushort) key, NetInfo.LaneType.Vehicle, VehicleInfo.VehicleType.Car, ref forwardLanes, ref backwardLanes);

                    XmlAttribute fLanes = mapDoc.CreateAttribute("FowardLanes");
                    fLanes.Value = forwardLanes.ToString();
                    segment.Attributes.Append(fLanes);

                    XmlAttribute bLanes = mapDoc.CreateAttribute("BackwardLanes");
                    bLanes.Value = backwardLanes.ToString();
                    segment.Attributes.Append(bLanes);

                    XmlElement startNode = this.nodeDictionary.ContainsKey(netSegment.m_startNode) ? this.nodeDictionary[netSegment.m_startNode] : (XmlElement)null;
                    XmlElement endNode = this.nodeDictionary.ContainsKey(netSegment.m_endNode) ? this.nodeDictionary[netSegment.m_endNode] : (XmlElement)null;
                    if (startNode == null)
                    {
                        DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "segment " + key.ToString() + netSegment.Info.name + " referes to node " + netSegment.m_startNode.ToString() + "(startNode) but that node is not created.");
                        ++key;
                        continue;
                    }
                    if (endNode == null)
                    {
                        DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "segment " + key.ToString() + netSegment.Info.name + " referes to node " + netSegment.m_endNode.ToString() + "(startNode) but that node is not created.");
                        ++key;
                        continue;
                    }

                    // Add element "StartLocation" as child to Segment, containing starting node id and coordinates
                    XmlElement startLoc = mapDoc.CreateElement("StartLocation");
                    XmlAttribute nodeId = mapDoc.CreateAttribute("NodeId");
                    nodeId.Value = netSegment.m_startNode.ToString();
                    startLoc.Attributes.Append(nodeId);
                    XmlAttribute x = mapDoc.CreateAttribute("x");
                    x.Value = startNode.ChildNodes[0].Attributes["x"].Value;
                    startLoc.Attributes.Append(x);
                    XmlAttribute y = mapDoc.CreateAttribute("y");
                    y.Value = startNode.ChildNodes[0].Attributes["y"].Value;
                    startLoc.Attributes.Append(y);
                    XmlAttribute z = mapDoc.CreateAttribute("z");
                    z.Value = startNode.ChildNodes[0].Attributes["z"].Value;
                    startLoc.Attributes.Append(z);
                    segment.AppendChild(startLoc);

                    // Do the same for the end node
                    XmlElement endLoc = mapDoc.CreateElement("EndLocation");
                    nodeId = mapDoc.CreateAttribute("NodeId");
                    nodeId.Value = netSegment.m_endNode.ToString();
                    endLoc.Attributes.Append(nodeId);
                    x = mapDoc.CreateAttribute("x");
                    x.Value = endNode.ChildNodes[0].Attributes["x"].Value;
                    endLoc.Attributes.Append(x);
                    y = mapDoc.CreateAttribute("y");
                    y.Value = endNode.ChildNodes[0].Attributes["y"].Value;
                    endLoc.Attributes.Append(y);
                    z = mapDoc.CreateAttribute("z");
                    z.Value = endNode.ChildNodes[0].Attributes["z"].Value;
                    endLoc.Attributes.Append(z);
                    segment.AppendChild(endLoc);

                    // Add new segment to as child to "Segments"
                    segments.AppendChild(segment);
                }
                ++key;
            }
            XmlElement root = mapDoc.DocumentElement;
            root.AppendChild(segments);
        }
    }
}
