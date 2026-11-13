"""Original ambient pad for the 27-second promo; no samples or percussion."""
import sys
import wave
import numpy as np

rate = 48000
seconds = 27
x = np.arange(rate * seconds) / rate
stereo = np.zeros((len(x), 2))
# Overlapping, slowly changing voicings avoid sharp note attacks.
for start, end, notes in [(0, 11, [48, 55, 62, 64]), (7, 20, [45, 52, 59, 60]), (16, 27, [48, 55, 60, 67])]:
    envelope = np.clip((x-start)/3, 0, 1) * np.clip((end-x)/4, 0, 1)
    envelope = np.sin(envelope*np.pi/2)**2
    for i, note in enumerate(notes):
        frequency = 440 * 2**((note-69)/12)
        for channel in range(2):
            detune = 1 + (channel*2-1)*0.0007
            phase = 2*np.pi*frequency*detune*x + i*0.7
            tone = np.sin(phase) + 0.12*np.sin(phase*2)
            stereo[:, channel] += tone * envelope * (0.035 if i else 0.022)
fade = np.sin(np.clip(x/2, 0, 1)*np.pi/2)**2
fade *= np.sin(np.clip((seconds-x)/3, 0, 1)*np.pi/2)**2
stereo *= fade[:, None]
assert np.isfinite(stereo).all() and np.abs(stereo).max() < 0.9
with wave.open(sys.argv[1], 'wb') as out:
    out.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
    out.writeframes((stereo*32767).astype('<i2').tobytes())
