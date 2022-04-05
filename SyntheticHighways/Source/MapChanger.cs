using System;
using System.Xml;
using System.IO;
using UnityEngine;
using ColossalFramework;
using ColossalFramework.Plugins;
using System.Diagnostics;
using ColossalFramework.Math;
using Microsoft.Win32;

namespace SyntheticHighways.MapChanger
{
    class MapChanger : MonoBehaviour
    {
        NetManager nManager;

        void Start()
        {
            // Initialize network manager
            nManager = Singleton<NetManager>.instance;
        }

        // Reinitializes the changes which were initially removed from the map
        public void ReinitChanges(string xml_fname)
        {
            // Read XML file to doc
            XmlDocument doc = new XmlDocument();
            string currDir = Directory.GetCurrentDirectory();
            string changePath = Path.Combine(currDir, xml_fname);
            doc.Load(changePath);

            // Make snapshot changes (changes to be made in between snapshots)
            // Remove the roads from the map for initial changes
            XmlElement root = doc.DocumentElement;
            XmlNodeList removeRoads = root.SelectNodes("/root/SnapshotChanges/RemoveRoad");
            foreach (XmlNode node in removeRoads)
            {
                RemoveRoad(Convert.ToUInt16(node.Attributes.GetNamedItem("SegmentId").Value));
            }

            // Add the roads from the map for initial changes
            XmlNodeList addRoads = root.SelectNodes("/root/SnapshotChanges/AddRoad");
            foreach (XmlNode node in addRoads)
            {
                ushort startNodeId = Convert.ToUInt16(node.Attributes.GetNamedItem("StartNodeId").Value);
                ushort endNodeId = Convert.ToUInt16(node.Attributes.GetNamedItem("EndNodeId").Value);
                uint prefabId = Convert.ToUInt16(node.Attributes.GetNamedItem("PrefabId").Value);
                AddRoad(startNodeId, endNodeId, prefabId);
            }

            // Make prefab changes in the map for initial changes
            XmlNodeList prefabChanges = root.SelectNodes("/root/SnapshotChanges/PrefabChange");
            /*foreach (XmlNode node in prefabChanges)
            {
                ushort segmentId = Convert.ToUInt16(node.Attributes.GetNamedItem("SegmentId").Value);
                ushort startNodeId = Convert.ToUInt16(node.Attributes.GetNamedItem("StartNodeId").Value);
                ushort endNodeId = Convert.ToUInt16(node.Attributes.GetNamedItem("EndNodeId").Value);
                uint prefabId = Convert.ToUInt16(node.Attributes.GetNamedItem("NewPrefabId").Value);
                PrefabChange(segmentId, startNodeId, endNodeId, prefabId);
            }*/

            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Snapshot changes have been made");

            // Delete the XML file containing the changes to be made
            /*try
            {
                if (File.Exists(changePath))
                {
                    File.Delete(changePath);
                }
                else
                {

                    DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Change suggestions file not found: " + changePath);
                }
            } catch (IOException ioExp)
            {
                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, ioExp.Message);
            }*/
        }

        public string MakeInitialChanges(XmlDocument mapDoc)
        {
            // Save xml to temporary file, which python script will read to have access to current road network
            string cityName = (string)((SimulationMetaData)Singleton<SimulationManager>.instance.m_metaData).m_CityName;
            string xmlFileName = cityName + "_temp.xml";
            mapDoc.Save(xmlFileName);

            // Start python process passing filename as argument
            ProcessStartInfo psi = new ProcessStartInfo();

            // Set arguments for python script to run
            // TODO: These paths should not be hardcoded
            psi.FileName = "C:\\Users\\KJW\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe";
            var script = @"""C:\Users\KJW\source\repos\SyntheticHighways\SyntheticHighways\Source\map_changer.py""";
            psi.Arguments = string.Format("{0} \"{1}\"", script, xmlFileName);
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;

            // Fetch results (results will contain the XML file path for changes to be made)
            var errors = "";
            var results = @"";
            try
            {
                using (var process = Process.Start(psi))
                {
                    errors = process.StandardError.ReadToEnd();
                    results = process.StandardOutput.ReadToEnd();
                }
            } catch (Exception e)
            {

                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, e.Message);
            }

            /*DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Errors: " + errors);
            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Results: " + results);*/
            // Delete temporary XML file now that python program has determined which roads to remove
            string currDir = Directory.GetCurrentDirectory();
            string deletePath = Path.Combine(currDir, xmlFileName);
            try
            {
                if (File.Exists(deletePath))
                {
                    File.Delete(deletePath);
                }
                else
                {

                    DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Change suggestions file not found: " + deletePath);
                }
            }
            catch (IOException ioExp)
            {
                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, ioExp.Message);
            }

            // Load in new XML document containing changes to be made
            XmlDocument doc = new XmlDocument();
            string changePath = Path.Combine(currDir, results);
            try
            {
                doc.Load(changePath);
            } catch( Exception e)
            {

                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, e.Message);
            }

            // Remove the roads from the map for initial changes
            XmlElement root = doc.DocumentElement;
            XmlNodeList removeRoads = root.SelectNodes("/root/InitialChanges/RemoveRoad");
            foreach (XmlNode node in removeRoads)
            {
                RemoveRoad(Convert.ToUInt16(node.Attributes.GetNamedItem("SegmentId").Value));
            }

            // Add the roads from the map for initial changes
            XmlNodeList addRoads = root.SelectNodes("/root/InitialChanges/AddRoad");
            foreach (XmlNode node in addRoads)
            {
                ushort startNodeId = Convert.ToUInt16(node.Attributes.GetNamedItem("StartNodeId").Value);
                ushort endNodeId = Convert.ToUInt16(node.Attributes.GetNamedItem("EndNodeId").Value);
                uint prefabId = Convert.ToUInt16(node.Attributes.GetNamedItem("PrefabId").Value);
                AddRoad(startNodeId, endNodeId, prefabId);
            }

            // Make prefab changes in the map for initial changes
            XmlNodeList prefabChanges = root.SelectNodes("/root/InitialChanges/PrefabChange");
            foreach (XmlNode node in prefabChanges)
            {
                ushort segmentId = Convert.ToUInt16(node.Attributes.GetNamedItem("SegmentId").Value);
                ushort startNodeId = Convert.ToUInt16(node.Attributes.GetNamedItem("StartNodeId").Value);
                ushort endNodeId = Convert.ToUInt16(node.Attributes.GetNamedItem("EndNodeId").Value);
                uint prefabId = Convert.ToUInt16(node.Attributes.GetNamedItem("NewPrefabId").Value);
                PrefabChange(segmentId, startNodeId, endNodeId, prefabId);
            }

            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Initial changes have been made");

            return changePath;
        }

        // Removes a road with a given ID
        void RemoveRoad(ushort segmentId)
        {
            nManager.ReleaseSegment(segmentId, false);
            Singleton<SimulationManager>.instance.m_currentBuildIndex -= 2u;
        }

        // Adds a new road between the given node IDs and prefabID (what kind of road to place)
        void AddRoad(ushort startNodeId, ushort endNodeId, uint prefabId)
        {
            ushort segmentId;
            Randomizer rand = new Randomizer(0u);

            NetInfo ni = PrefabCollection<NetInfo>.GetPrefab(prefabId);

            Vector3 pos1 = nManager.m_nodes.m_buffer[(int)startNodeId].m_position;
            Vector3 pos2 = nManager.m_nodes.m_buffer[(int)endNodeId].m_position;
            Vector3 vec = pos2 - pos1;
            vec = VectorUtils.NormalizeXZ(vec);
            nManager.CreateSegment(out segmentId, ref rand, ni, startNodeId, endNodeId, vec, -vec, Singleton<SimulationManager>.instance.m_currentBuildIndex, Singleton<SimulationManager>.instance.m_currentBuildIndex, false);
            Singleton<SimulationManager>.instance.m_currentBuildIndex += 2u;
        }

        void PrefabChange(ushort segmentId, ushort startNodeId, ushort endNodeId, uint prefabId)
        {
            RemoveRoad(segmentId);
            AddRoad(startNodeId, endNodeId, prefabId);
        }
    }
}
