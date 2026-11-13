"""Generate an original 45-second synthesized arpeggio; no sampled audio."""
import sys, wave
import numpy as np
sr=48000
a=np.zeros(sr*45)
chords=[[48,55,60,64],[45,52,57,60],[41,48,53,57],[43,50,55,59]]
for k in range(90):
    start=int(k*.5*sr)
    note=chords[(k//16)%4][k%4]+12
    freq=440*2**((note-69)/12)
    t=np.arange(min(2*sr,len(a)-start))/sr
    envelope=(1-np.exp(-t*45))*np.exp(-t*2.4)
    a[start:start+len(t)]+=(np.sin(2*np.pi*freq*t)+.18*np.sin(4*np.pi*freq*t))*envelope*.065
fade=np.minimum(np.arange(len(a))/sr/2,1)*np.minimum((len(a)-np.arange(len(a)))/sr/3,1)
a*=fade
assert np.isfinite(a).all() and np.max(np.abs(a))<1
with wave.open(sys.argv[1],'wb') as w:
    w.setparams((1,2,sr,0,'NONE','not compressed'))
    w.writeframes((a*32767).astype('<i2').tobytes())
