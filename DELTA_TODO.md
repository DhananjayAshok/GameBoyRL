4. `propose_zeroshot.py`


5. Make a VLM script that:
    - Starts in init state 
    - Given a task string
    - Tries to execute it with executor and judges completion on its own
    - Regardless of what it thinks, judge it for completion. If success then add as a task with trajectory in pickle form. 
        - Else, critique the trajectory and identify whether its impossible or to do the task and if possible, then give a hint that could have helped fix the trajectory and try again. 
        - K loops of this, any success is good. 
    - DONE (TEST)

6. Make a VLM script that:
    - Starts in init state
    - VLM static inference: Comes up with things it can explore (focus on mechanisms and areas of interest in your local neighbourhood, not other regions entirely)
    - Pick an exploration path or thing to explore
    - Explore what happens by an ExplorationSupervisor that gives orders to an executor and parses the observation streams to build:
        - possibly interesting secondary states that one could go to (e.g. location, specific menu, specific agent state, new situation, or something else)
    - Combine SupervisorReports into a consolidated understanding of all of these to get possible tasks.
    - Try to reach each secondary states with an Execution call and judge success
    - If success, then save state and trajectory. You need to be able to stitch this trajectory with others easily. [CURIOSITY INTEGRATION REMAINING]


7. Ensure you can call curiosity exploration on these secondary states with a small step budget and curiosity buffer of the initial run. Be able to group, and save with a stitch at the end. [REMAINING]

8. Have a VLM script that:
    - Given an initial state and a task to achieve with a single trajectory
    - Extracts a gold trajectory guidance on how to do it
    - DONE (TEST)

9. Have a VLM script that give:
    - init state, task, gold guidance
    - Tries with an executor
    - Judges its own success using the same mechanism as above. 
    - If success, save to one
    - If fail, save to a different one, with option of not saving. DONE (TEST)

10. Set up VLM finetuning to plug this in and train on the successful trajectories. 