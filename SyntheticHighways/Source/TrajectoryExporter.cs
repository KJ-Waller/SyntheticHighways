using System.Xml;
using UnityEngine;
using ColossalFramework.Plugins;
using ColossalFramework;
using System.Collections;
using System.IO;
using System.Collections.Generic;

namespace SyntheticHighways.TrajectoryExporter
{
    class TrajectoryExporter : MonoBehaviour 
    {
        VehicleManager vManager;
        SimulationManager sManager;
        XmlDocument vehDoc;
        private Dictionary<int, List<Vehicle>> vehDictionary;
        private Dictionary<int, List<string>> vehTimestamps;

        void Start()
        {
            // Initialize vehicle manager
            vManager = Singleton<VehicleManager>.instance;
            sManager = Singleton<SimulationManager>.instance;
        }

        public IEnumerator StartExport(int snapNumber, float timeInterval, int repetitions)
        {
            // Initialize dictionaries for keeping track of vehicle locations and timestamps
            vehDictionary = new Dictionary<int, List<Vehicle>>();
            vehTimestamps = new Dictionary<int, List<string>>();

            // Start Coroutine which calls ExportTrajectories every x seconds
            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Initializing Coroutine");
            yield return StartCoroutine(ExportTrajectories(snapNumber, timeInterval, repetitions));
        }

        IEnumerator ExportTrajectories(int snapNumber, float timeInterval, int repetitions)
        {
            // Fetch "repetitions" number of locations per vehicle, waiting "timeInterval" number of seconds in between
            for (int i = 0; i < repetitions; i++)
            {
                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Trajectories collection #" + i.ToString());
                ExportVehicleLocations();
                yield return new WaitForSeconds(timeInterval);
            }

            // Initialize XML document
            vehDoc = new XmlDocument();
            XmlDeclaration xmlDeclaration = vehDoc.CreateXmlDeclaration("1.0", "UTF-8", null);
            vehDoc.AppendChild(xmlDeclaration);
            XmlElement root = vehDoc.CreateElement("Trajectories");
            vehDoc.AppendChild(root);

            // Initialize vehicles element which will contain all vehicles
            XmlElement vehicles = vehDoc.CreateElement("Vehicles");

            foreach (KeyValuePair<int, List<Vehicle>> vehEntry in vehDictionary)
            {
                // Create a vehicle element with some basic information about the vehicle
                int key = vehEntry.Key;
                List<Vehicle> locations = vehEntry.Value;

                VehicleInfo vehInfo = locations[0].Info;

                XmlElement vehicleElement = vehDoc.CreateElement("Vehicle");
                XmlAttribute id = vehDoc.CreateAttribute("id");
                id.Value = key.ToString();
                vehicleElement.Attributes.Append(id);

                XmlAttribute vehServ = vehDoc.CreateAttribute("Service");
                vehServ.Value = vehInfo.GetService().ToString();
                vehicleElement.Attributes.Append(vehServ);

                XmlAttribute vehSubServ = vehDoc.CreateAttribute("SubService");
                vehSubServ.Value = vehInfo.GetSubService().ToString();
                vehicleElement.Attributes.Append(vehSubServ);

                XmlAttribute vehType = vehDoc.CreateAttribute("VehicleType");
                vehType.Value = vehInfo.m_vehicleType.ToString();
                vehicleElement.Attributes.Append(vehType);

                XmlAttribute vehTypeGet = vehDoc.CreateAttribute("VehicleTypeGet");
                vehTypeGet.Value = vehInfo.GetType().ToString();
                vehicleElement.Attributes.Append(vehTypeGet);

                XmlAttribute vehTag = vehDoc.CreateAttribute("Tag");
                vehTag.Value = vehInfo.tag.ToString();
                vehicleElement.Attributes.Append(vehTag);


                // For each location, append a location element to the vehicle element in the XML doc
                int locIdx = 0;
                foreach (Vehicle veh in locations)
                {

                    // Get position and velocity of vehicle in last frame
                    Vector3 pos = veh.GetLastFramePosition();
                    Vector3 vel = veh.GetLastFrameVelocity();

                    // Add location element with x,y,z attributes to vehicle element
                    XmlElement loc = vehDoc.CreateElement("Location");
                    XmlAttribute x = vehDoc.CreateAttribute("x");
                    x.Value = pos.x.ToString();
                    loc.Attributes.Append(x);
                    XmlAttribute y = vehDoc.CreateAttribute("y");
                    y.Value = pos.y.ToString();
                    loc.Attributes.Append(y);
                    XmlAttribute z = vehDoc.CreateAttribute("z");
                    z.Value = pos.z.ToString();
                    loc.Attributes.Append(z);

                    // Add speed to location element
                    XmlAttribute speed = vehDoc.CreateAttribute("speed");
                    speed.Value = vel.magnitude.ToString();
                    loc.Attributes.Append(speed);

                    // Add speed to location element
                    XmlAttribute timestamp = vehDoc.CreateAttribute("timestamp");
                    timestamp.Value = vehTimestamps[key][locIdx];
                    loc.Attributes.Append(timestamp);

                    // Add heading child element to location element with n1,n2,n3 as attributes (normalized x,y,z)
                    XmlElement heading = vehDoc.CreateElement("Heading");
                    XmlAttribute n1 = vehDoc.CreateAttribute("n1");
                    n1.Value = vel.normalized.x.ToString();
                    heading.Attributes.Append(n1);
                    XmlAttribute n2 = vehDoc.CreateAttribute("n2");
                    n2.Value = vel.normalized.y.ToString();
                    heading.Attributes.Append(n2);
                    XmlAttribute n3 = vehDoc.CreateAttribute("n3");
                    n3.Value = vel.normalized.z.ToString();
                    heading.Attributes.Append(n3);
                    loc.AppendChild(heading);

                    // Add location element as child to the vehicle
                    vehicleElement.AppendChild(loc);

                    locIdx++;
                }

                // Append vehicle to vehicles
                vehicles.AppendChild(vehicleElement);

            }

            root = vehDoc.DocumentElement;
            root.AppendChild(vehicles);

            // After getting all trajectories, save the XML file
            // Create a folder to save XML to
            string cityName = (string)((SimulationMetaData)Singleton<SimulationManager>.instance.m_metaData).m_CityName;
            string currDir = Directory.GetCurrentDirectory();
            string folder = Path.Combine(currDir, "SyntheticHighways");

            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }

            // Save trajectories to XML file in said folder
            string fname = cityName + "_" + snapNumber.ToString() + "_trajectories.xml";
            string saveName = Path.Combine(folder, fname);
            vehDoc.Save(saveName);
            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Trajectories saved to: " + saveName);
        }

        void ExportVehicleLocations()
        {
            // Pause simulation while locations are being fetched
            /*sManager.SimulationPaused = true;*/


            // For each vehicle in the buffer, save a location attribute
            int key = 0;
            foreach (Vehicle veh in (Vehicle[])((Array16<Vehicle>)vManager.m_vehicles).m_buffer)
            {

                if (veh.m_flags != 0)
                {
                    
                    List<Vehicle> locations;
                    List<string> timestamps;

                    // Check if key exists in the dictionaries, otherwise, initilize new lists
                    if (vehDictionary.ContainsKey(key))
                    {
                        locations = vehDictionary[key];
                        timestamps = vehTimestamps[key];
                    } else
                    {
                        locations = new List<Vehicle>();
                        timestamps = new List<string>();
                    }

                    // Add the new vehicle location and current timestamp to dictionary entry
                    locations.Add(veh);
                    timestamps.Add(sManager.m_currentGameTime.ToString());
                    vehTimestamps[key] = timestamps;
                    vehDictionary[key] = locations;
                }

                ++key;
            }
        }
    }
}
