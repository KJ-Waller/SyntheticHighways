
using UnityEngine;
using ICities;
using ColossalFramework.Plugins;
using System.Xml;
using SyntheticHighways.TrajectoryExporter;
using SyntheticHighways.MapExporter;
using SyntheticHighways.MapChanger;
using SyntheticHighways.MapLoader;
using System.Collections;
using ColossalFramework;

namespace SyntheticHighways
{


    public class SyntheticHighwaysMain : IUserMod
    {
        public string Name
        {
            get { return "Synthetic Highways"; }
        }

        public string Description
        {
            get { return "Exports synthetic GPS/probe traces including road network"; }
        }

        public void OnSettingsUI(UIHelperBase helper)
        {
            // Load configuration
            SyntheticHighwaysConfiguration config = Configuration<SyntheticHighwaysConfiguration>.Load();

            // Add text fields for mod start delay, trajectory time interval, and max trajectory length
            helper.AddTextfield("Mod Start Delay", config.ModStartDelay.ToString(), (value) =>
            {
                Debug.Log(value.ToString());
                config.ModStartDelay = int.Parse(value);
                Configuration<SyntheticHighwaysConfiguration>.Save();
            });

            helper.AddTextfield("Trajectory Time Interval", config.TrajectoryTimeInterval.ToString(), (value) =>
            {
                Debug.Log(value.ToString());
                config.TrajectoryTimeInterval = float.Parse(value);
                Configuration<SyntheticHighwaysConfiguration>.Save();
            });

            helper.AddTextfield("Max Trajectory Length", config.MaxTrajectoryLength.ToString(), (value) =>
            {
                Debug.Log(value.ToString());
                config.MaxTrajectoryLength = int.Parse(value);
                Configuration<SyntheticHighwaysConfiguration>.Save();
            });

            helper.AddTextfield("Number of Batches", config.BatchNumber.ToString(), (value) =>
            {
                Debug.Log(value.ToString());
                config.BatchNumber = int.Parse(value);
                Configuration<SyntheticHighwaysConfiguration>.Save();
            });
        }
    }

    public class Loader : LoadingExtensionBase
    {
        public override void OnLevelLoaded(LoadMode mode)
        {
            GameObject mod = new GameObject("SyntheticHighwaysMod");
            mod.AddComponent<SyntheticHighwaysBehaviour>();
        }
    }

    public class SyntheticHighwaysBehaviour : MonoBehaviour
    {

        float trajectoryTimeInterval = 10f;
        int maxTrajectoryLength = 5;
        int modStartDelay = 5;
        int batchNumber = 3;

        bool modRunning = false;

        GameObject mapExpGO = new GameObject("MapExporterObj");
        MapExporter.MapExporter mapExporter;

        GameObject trajExpGO = new GameObject("TrajectoryExporterObj");
        TrajectoryExporter.TrajectoryExporter trajExporter;

        GameObject mapChangerGO = new GameObject("MapChangerObj");
        MapChanger.MapChanger mapChanger;

        GameObject mapLoaderGO = new GameObject("MapLoaderObj");
        MapLoader.MapLoader mapLoader;

        SimulationManager simMan;

        void Start()
        {
            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Synthetic Highways mod is running");

            // Get mod configurations from config file
            SyntheticHighwaysConfiguration config = Configuration<SyntheticHighwaysConfiguration>.Load();
            trajectoryTimeInterval = config.TrajectoryTimeInterval;
            maxTrajectoryLength = config.MaxTrajectoryLength;
            modStartDelay = config.ModStartDelay;
            batchNumber = config.BatchNumber;

            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Mod will start in: " + modStartDelay.ToString() + " seconds");
            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Time Interval: " + trajectoryTimeInterval.ToString());
            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Trajectory Length: " + maxTrajectoryLength.ToString());
            DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Number of batches: " + batchNumber.ToString());

            // Initialize map exporter
            mapExporter = mapExpGO.AddComponent<MapExporter.MapExporter>();
            trajExporter = trajExpGO.AddComponent<TrajectoryExporter.TrajectoryExporter>();
            mapChanger = mapChangerGO.AddComponent<MapChanger.MapChanger>();
            mapLoader = mapLoaderGO.AddComponent<MapLoader.MapLoader>();

            // Initialize simulation manager
            simMan = Singleton<SimulationManager>.instance;

            // Make sure simulation isn't paused
            if (simMan.SimulationPaused)
            {
                simMan.SimulationPaused = false;
            }

            // Start collecting data
            StartCoroutine(CollectData());
        }

        IEnumerator CollectData()
        {
            /*modRunning = true;*/
            // Wait before starting the mod
            yield return new WaitForSecondsRealtime(modStartDelay);

            // Export the map to XML for Python change suggestor (saved to temporary file)
            XmlDocument mapDoc = mapExporter.ExportMap(1, false);

            // Make initial changes (remove some roads)
            string temp_fname = mapChanger.MakeInitialChanges(mapDoc);

            // Wait for changes to take effect.
            yield return new WaitForSecondsRealtime(20);

            // Export the map to XML for first snapshot
            mapExporter.ExportMap(1, true);
            for (int i = 0; i < batchNumber; i++)
            {
                // Record trajectories for first snapshot
                yield return StartCoroutine(trajExporter.StartExport(1, trajectoryTimeInterval, maxTrajectoryLength, i));
            }

            // Add in removed roads again
            mapChanger.ReinitChanges(temp_fname);
            // TODO: Other changes
            // * Add lanes
            // * Change road type
            // * Change road directionality

            // Wait for changes to take effect.
            yield return new WaitForSecondsRealtime(20);

            for (int i = 0; i < batchNumber; i++)
            {
                // Export map for second snapshot
                mapExporter.ExportMap(2, true);
                // Record trajectories for second snapshot
                yield return StartCoroutine(trajExporter.StartExport(2, trajectoryTimeInterval, maxTrajectoryLength, i));
            }

            LoadNextMap();
        }

        void LoadNextMap()
        {
            // Load the next map
            mapLoader.LoadNextMap();
        }


        void Update()
        {
            // Make sure simulation isn't paused
            if (simMan.SimulationPaused & !modRunning)
            {
                simMan.SimulationPaused = false;
            }

        }

    }

}
