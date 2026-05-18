using System.Collections;
using UnityEngine;
using UnityEngine.UI; // <-- REQUIRED for the Image component
using TMPro;

[System.Serializable]
public class ShoulderMovement
{
    public int movementID; 
    public string movementName;
    public float[] targetAngles = new float[4]; 
}

public class MovementPromptManager : MonoBehaviour
{
    [Header("UI Elements")]
    public TMP_Text promptText;
    public Image timerRing; // <-- Drag your new TimerRing UI Image here

    [Header("Connections")]
    public UDPProstheticController prostheticController;

    [Header("Target Prosthetic Joints")]
    public Transform targetShoulderYaw;
    public Transform targetShoulderPitch;
    public Transform targetShoulderRoll; 
    public Transform targetElbowFlexion;

    [Header("Target Visuals (For Color Changing)")]
    public GameObject targetProstheticRoot; 
    private Renderer[] targetRenderers;

    [Header("Timing Settings")]
    public float restDuration = 2.0f;
    public float movementTimeAllowance = 3.0f; 
    public float holdDuration = 3.0f;          

    [Header("Movements")]
    public ShoulderMovement[] movements; 

    private float yellowThreshold = 20.0f;
    private float greenThreshold = 5.0f;
    private bool isTesting = false;

    void Start()
    {
        if (targetProstheticRoot != null)
            targetRenderers = targetProstheticRoot.GetComponentsInChildren<Renderer>();

        if (prostheticController != null)
            prostheticController.OnRecordCommandReceived += StartTrial;

        SetUIWaiting();
    }

    private void StartTrial(int incomingMoveId)
    {
        if (!isTesting)
        {
            StartCoroutine(MovementTestRoutine(incomingMoveId));
        }
    }

    private IEnumerator MovementTestRoutine(int moveId)
    {
        isTesting = true;
        ShoulderMovement currentTask = GetMovementByID(moveId);
        ShoulderMovement restTask = GetMovementByID(9); 

        if (currentTask == null || restTask == null)
        {
            Debug.LogError($"Movement ID {moveId} or Rest ID 9 not found in the Inspector array!");
            isTesting = false;
            yield break;
        }

        // --- PRE-TRIAL: Sync with Python's 3-second countdown ---
        SetTargetPose(restTask.targetAngles);
        SetTargetColor(Color.white);
        
        for (int i = 3; i > 0; i--)
        {
            promptText.text = $"Syncing...\n{i}";
            promptText.color = Color.yellow;
            
            // Fill the ring fully in white during sync
            if (timerRing != null) { timerRing.fillAmount = 1f; timerRing.color = Color.white; }
            
            yield return new WaitForSeconds(1.0f);
        }

        // --- THE 10 REPETITIONS LOOP ---
        int totalRepetitions = 10;
        for (int rep = 1; rep <= totalRepetitions; rep++)
        {
            // --- PHASE 1: Resting ---
            float timer = 0f;
            while (timer < restDuration)
            {
                promptText.text = $"Rep {rep}/{totalRepetitions}\nResting Condition\nRelax your arm.";
                promptText.color = Color.white;
                UpdateTimerVisual(timer, restDuration);
                timer += Time.deltaTime;
                yield return null;
            }

            // --- PHASE 2: Elevation (Move to pose) ---
            timer = 0f;
            SetTargetPose(currentTask.targetAngles); 
            while (timer < movementTimeAllowance)
            {
                promptText.text = $"Rep {rep}/{totalRepetitions}\nMove to:\n<b>{currentTask.movementName}</b>";
                UpdateTargetColor(currentTask);
                UpdateTimerVisual(timer, movementTimeAllowance);
                timer += Time.deltaTime;
                yield return null; 
            }

            // --- PHASE 3: Isometric Holding ---
            timer = 0f;
            while (timer < holdDuration)
            {
                promptText.text = $"Rep {rep}/{totalRepetitions}\nHOLD:\n<b>{currentTask.movementName}</b>";
                UpdateTargetColor(currentTask);
                UpdateTimerVisual(timer, holdDuration);
                timer += Time.deltaTime;
                yield return null;
            }

            // --- PHASE 4: Return to Rest ---
            timer = 0f;
            SetTargetPose(restTask.targetAngles); 
            SetTargetColor(Color.white);
            while (timer < restDuration)
            {
                promptText.text = $"Rep {rep}/{totalRepetitions}\nReturn to Rest\nRelax your arm.";
                promptText.color = Color.white;
                UpdateTimerVisual(timer, restDuration);
                timer += Time.deltaTime;
                yield return null;
            }
        }

        // --- END TRIAL: Send Stop Command to Python ---
        prostheticController.SendCommandToPython("CMD:STOP");
        SetUIWaiting();
        isTesting = false;
    }

    // --- NEW HELPER METHOD: Handles the Circular Loading Bar ---
    private void UpdateTimerVisual(float currentTimer, float totalDuration)
    {
        if (timerRing == null) return;

        // Calculate how far along we are (0.0 to 1.0)
        float completionRatio = currentTimer / totalDuration;
        timerRing.fillAmount = completionRatio;

        // Color Logic: Green (0-33%), Yellow (33-66%), Red (66-100%)
        if (completionRatio <= 0.33f)
            timerRing.color = Color.green;
        else if (completionRatio <= 0.66f)
            timerRing.color = Color.yellow;
        else
            timerRing.color = Color.red;
    }

    private ShoulderMovement GetMovementByID(int id)
    {
        foreach (var move in movements)
        {
            if (move.movementID == id) return move;
        }
        return null;
    }

    private void SetUIWaiting()
    {
        promptText.text = "Waiting for Python Controller...";
        promptText.color = Color.gray;
        SetTargetColor(new Color(1, 1, 1, 0)); 
        
        if (timerRing != null) 
        {
            timerRing.fillAmount = 0f; // Hide ring while waiting
        }
    }

    private void SetTargetPose(float[] angles)
    {
        if (targetShoulderYaw != null) targetShoulderYaw.localRotation = Quaternion.AngleAxis(angles[0], Vector3.forward);
        if (targetShoulderPitch != null) targetShoulderPitch.localRotation = Quaternion.AngleAxis(angles[1], Vector3.left);
        if (targetShoulderRoll != null) targetShoulderRoll.localRotation = Quaternion.AngleAxis(angles[2], Vector3.up);
        if (targetElbowFlexion != null) targetElbowFlexion.localRotation = Quaternion.AngleAxis(angles[3], Vector3.forward);
    }

    private void UpdateTargetColor(ShoulderMovement task)
    {
        if (prostheticController == null || targetRenderers == null) return;
        float[] currentAngles = prostheticController.GetCurrentAngles(); 
        float maxError = 0f;
        for (int i = 0; i < 4; i++)
        {
            float error = Mathf.Abs(Mathf.DeltaAngle(currentAngles[i], task.targetAngles[i]));
            if (error > maxError) maxError = error;
        }
        Color colorToApply = Color.red;
        if (maxError <= greenThreshold) colorToApply = Color.green;
        else if (maxError <= yellowThreshold) colorToApply = Color.yellow;
        SetTargetColor(colorToApply);
    }

    private void SetTargetColor(Color newColor)
    {
        if (targetRenderers == null) return;
        foreach (Renderer r in targetRenderers)
        {
            if (r != null)
            {
                foreach (Material mat in r.materials) mat.color = newColor;
            }
        }
    }
}