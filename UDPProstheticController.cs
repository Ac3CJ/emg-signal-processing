using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System;
using System.Globalization;

public class UDPProstheticController : MonoBehaviour
{
    [Header("Network Settings")]
    public int port = 5005;

    [Header("Joint Transforms (Proxy Hierarchy)")]
    public Transform shoulderYaw;   
    public Transform shoulderPitch; 
    public Transform shoulderRoll;  
    public Transform elbowFlexion;  

    [Header("Dynamics Simulation")]
    public float smoothTime = 0.1f; 

    // --- NEW: Event to trigger the Prompt Manager ---
    public event Action<int> OnRecordCommandReceived;

    private UdpClient udpClient;
    private Thread receiveThread;
    private bool isRunning = false;

    private float[] targetAngles = new float[4]; 
    private float[] currentAngles = new float[4]; 
    private float[] jointVelocities = new float[4]; 

    // --- NEW: Thread-safe command handling & Networking ---
    private IPEndPoint piEndPoint; 
    private volatile bool hasPendingCommand = false;
    private int pendingMoveId = -1;

    void Start()
    {
        Application.runInBackground = true;
        StartUDPListener();
    }

    private void StartUDPListener()
    {
        try
        {
            udpClient = new UdpClient(port);
            isRunning = true;
            receiveThread = new Thread(new ThreadStart(ReceiveData));
            receiveThread.IsBackground = true;
            receiveThread.Start();
            Debug.Log($"[Prosthetic Twin] Listening on port {port}...");
        }
        catch (Exception e) { Debug.LogError($"UDP Error: {e.Message}"); }
    }

    private void ReceiveData()
    {
        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
        while (isRunning)
        {
            try
            {
                byte[] data = udpClient.Receive(ref anyIP);
                
                // Save the Pi's IP, but force the port to 5005 so our STOP command hits your Python listener
                piEndPoint = new IPEndPoint(anyIP.Address, port);

                string text = Encoding.UTF8.GetString(data);

                Debug.Log($"[UDP RECEIVE RAW]: {text}");
                
                // --- INTERCEPT COMMANDS ---
                if (text.StartsWith("CMD:"))
                {
                    if (text.StartsWith("CMD:GO_RECORDING:"))
                    {
                        string[] parts = text.Split(':');
                        if (parts.Length == 3 && int.TryParse(parts[2], out int moveId))
                        {
                            pendingMoveId = moveId;
                            hasPendingCommand = true; // Flag for the main thread
                        }
                    }
                }
                // --- PARSE KINEMATICS ---
                else
                {
                    string[] parts = text.Split(',');
                    if (parts.Length >= 4)
                    {
                        float.TryParse(parts[0], out targetAngles[0]);
                        float.TryParse(parts[1], out targetAngles[1]);
                        float.TryParse(parts[2], out targetAngles[2]);
                        float.TryParse(parts[3], out targetAngles[3]);
                    }
                }
            }
            catch (Exception e) { if (isRunning) Debug.LogWarning($"UDP Receive Error: {e.Message}"); }
        }
    }

    void Update()
    {
        // Safely trigger the event on Unity's main thread
        if (hasPendingCommand)
        {
            hasPendingCommand = false;
            // ACK the Pi so its 3s countdown lines up with our "Syncing..." countdown.
            SendCommandToPython("CMD:ACK");
            OnRecordCommandReceived?.Invoke(pendingMoveId);
        }

        if (shoulderYaw != null) ApplySmoothRotation(shoulderYaw, 0, targetAngles[0], Vector3.forward);
        if (shoulderPitch != null) ApplySmoothRotation(shoulderPitch, 1, targetAngles[1], Vector3.left);
        if (shoulderRoll != null) ApplySmoothRotation(shoulderRoll, 2, targetAngles[2], Vector3.up);
        if (elbowFlexion != null) ApplySmoothRotation(elbowFlexion, 3, targetAngles[3], Vector3.forward);
    }

    private void ApplySmoothRotation(Transform joint, int index, float targetAngle, Vector3 axis)
    {
        currentAngles[index] = Mathf.SmoothDampAngle(currentAngles[index], targetAngle, ref jointVelocities[index], smoothTime);
        joint.localRotation = Quaternion.AngleAxis(currentAngles[index], axis);
    }

    // --- NEW: Send commands back to the Python Script ---
    public void SendCommandToPython(string command)
    {
        if (piEndPoint != null && udpClient != null)
        {
            byte[] data = Encoding.UTF8.GetBytes(command);
            udpClient.Send(data, data.Length, piEndPoint);
            Debug.Log($"[Prosthetic Twin] Sent command to Pi: {command}");
        }
        else
        {
            Debug.LogWarning("Cannot send command. No connection established with Pi yet.");
        }
    }

    private void OnApplicationQuit() { CleanUp(); }
    private void OnDestroy() { CleanUp(); }
    private void CleanUp()
    {
        isRunning = false;
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort();
        if (udpClient != null) udpClient.Close();
    }

    public float[] GetCurrentAngles() { return currentAngles; }
}