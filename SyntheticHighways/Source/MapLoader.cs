using System;
using System.IO;
using UnityEngine;
using ColossalFramework;
using ColossalFramework.Plugins;
using Package = ColossalFramework.Packaging.Package;
using UIView = ColossalFramework.UI.UIView;

namespace SyntheticHighways.MapLoader
{
    class MapLoader : MonoBehaviour
    {

        string[] mapAssetNames;

        void Start()
        {
            // TODO: Get savegame directory programatically
            mapAssetNames = Directory.GetFiles("C:\\Users\\KJW\\AppData\\Local\\Colossal Order\\Cities_Skylines\\Saves\\");
        }

        // Loads next map in the list
        public void LoadNextMap()
        {
            // Get city name
            string cityName = Singleton<SimulationManager>.instance.m_metaData.m_CityName;

            // Check which index this city associates to in the savegame files
            int saveGameIdx = -1;
            for (int i = 0; i < mapAssetNames.Length; i++)
            {
                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Does " + mapAssetNames[i] + " contain" + cityName);
                if (IsSameCity(mapAssetNames[i], cityName))
                {

                    DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Yes it does!");
                    saveGameIdx = i;
                }
            }

            // Load the next map in the list
            if (saveGameIdx == -1)
            {
                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "City name not found: " + cityName.ToString());
                throw new FileNotFoundException(cityName);
            }
            else if (saveGameIdx == (mapAssetNames.Length - 1))
            {
                DebugOutputPanel.AddMessage(PluginManager.MessageType.Message, "Last map already loaded");
            }
            else
            {
                LoadMap(saveGameIdx + 1);
            }
        }

        // Checks if a given savegame filename and cityname associate to the same map
        bool IsSameCity(string fname, string cityName)
        {
            var package = GetSaveGameFromPath(fname);
            var savegameMetaData = GetMetaDataFromPackage(package);

            if (savegameMetaData.cityName == cityName)
            {
                return true;
            } else
            {
                return false;
            }
        }

        void LoadMap(int idx)
        {

            string fname = mapAssetNames[idx];

            // Ensure that the LoadingManager is ready. Don't know if thats really necessary but doesn't hurt either.
            Singleton<LoadingManager>.Ensure();

            // saveName should be the path to the file (full qualified, including save file extension)
            var package = GetSaveGameFromPath(fname);
            var savegameMetaData = GetMetaDataFromPackage(package);

            var metaData = new SimulationMetaData()
            {
                m_CityName = savegameMetaData.cityName,
                m_updateMode = SimulationManager.UpdateMode.LoadGame,
                m_environment = UIView.GetAView().panelsLibrary.Get<LoadPanel>("LoadPanel").m_forceEnvironment
            };

            Singleton<LoadingManager>.instance.LoadLevel(savegameMetaData.assetRef, "Game", "InGame", metaData, false);
        }

        private Package GetSaveGameFromPath(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException(path);

            var package = new Package(Path.GetFileNameWithoutExtension(path), path);

            return package;
        }

        private SaveGameMetaData GetMetaDataFromPackage(Package package)
        {
            if (package == null)
                throw new ArgumentNullException(package.packageName);

            var asset = package.Find(package.packageMainAsset);
            var metaData = asset.Instantiate<SaveGameMetaData>();

            return metaData;
        }
    }
}
