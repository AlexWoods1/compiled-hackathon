# Core idea

A local voice filter for a live call, such as Zoom. It censors spoken secrets on the way out, so the other people on the call never hear them.

The user picks this app as their Zoom microphone. They supply a topics file: one sensitive concept per line. The app takes the real mic, transcribes speech on device, and checks each short chunk against that file. A match is muted or replaced with a beep. Everything else is passed to Zoom.

The check is a lookup, not a conversation. Exact patterns (passwords, keys, account numbers) match first. The rest is embedding similarity: each topic line is embedded once at startup, and each transcript chunk is compared to those vectors. A small language model is only a second pass on borderline matches.

Speech stays local. A short delay buffer holds the outgoing audio long enough to cover the words that matched, and short enough that the call still feels live.
