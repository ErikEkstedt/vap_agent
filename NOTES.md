# VAP Agent Notes


* [Overbleed is recognized with context]()



# Overbleed is recognized with context

![vad requires context](assets/note_1_vad.png)

The overbleed in channel 2 is recognized as noise after around a second of speech. We note that the entropy is decreasing over the same period (kinda).

# Entropy at onset

![vad requires context](assets/note_2_first_word_entropy.png)

The entropy "generally" increases at complicated steps (e.g. TRP, a turn-shift is likely ). However, when we get into a new utterance/turn we have some certainty regarding the future (e.g. the projection-window). That certainty is greater over the **first** word but "quickly" increases at the start of the **second**.

# Entropy at turn-shift projections

![vad requires context](assets/note_3_turn_shift_projection_entropy.png)

The entropy "generally" increases at the end of turns which can indicate that the model anticipates a turn-shift. During a turn the model can mostly focus on modelling the current speaker but in proximity to a turn-shift the influence of the second speaker increases which is reflected in the rise in entropy.
