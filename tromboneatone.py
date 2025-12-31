import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
import mouse
import math
import time
import threading

# Settings/configuration
INVERTED_CONTROLS = True
MIDDLE_OCTAVE = 4 # 4 means C5 is the middle note
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
SAMPLE_FREQ = 44100 # sample frequency in Hz
LOWEST_SUPPORTED_FREQUENCY = 246.9 # 246.9 is B3, the lowest note playable
USE_INTERPOLATION = True
POWER_THRESH = 0.08 # average amplitude cutoff for pitch detection

# Debug settings
PRINT_NOTE = False
PRINT_CB_DATA = False
MOUSE_ACTIVE = True   #if true, mouse will move to the note

# could try playing at a higher octave for lower latency
# seems like most of the latency is coming from the game; it's very similar between when I use the Otamatone and the mouse
# so I might be able to configure all of that to work well, between audio and video latency
# and not waste time trying to optimize my latency here any more
# I could also turn the trombone volume off if that's where some of the perceived latency is coming from

BLOCK_LENGTH_SECONDS = 2 / LOWEST_SUPPORTED_FREQUENCY
BLOCK_SIZE = int(BLOCK_LENGTH_SECONDS * SAMPLE_FREQ) #samples per window
MAX_FREQUENCY = 1500

bFreqTable= [30.8677,61.7354,123.471,246.942,493.883,987.767,1975.53]
#cFreqTable= [32.703,65.406,130.813,261.626,523.251,1046.5,2093.0]
cSharpFreqTable= [34.6478,69.2957,138.591,277.183,554.365,1108.73,2217.46]
logMinFreq = math.log(bFreqTable[MIDDLE_OCTAVE-1], 2)
logMaxFreq = math.log(cSharpFreqTable[MIDDLE_OCTAVE+1], 2)
logFreqRange = logMaxFreq - logMinFreq
def getPitchTrombonePercent(frequency) -> float:
  return np.clip((math.log(frequency, 2) - logMinFreq) / logFreqRange, 0, 1)

def getScreenPoint(pitchPercent):
  bottomMarginSize = 108 # TODO proportional to resolution
  topMarginSize = 109
  gameHeightInputSize = SCREEN_HEIGHT - bottomMarginSize - topMarginSize
  if not INVERTED_CONTROLS:
    pitchPercent = 1 - pitchPercent
  y = int(gameHeightInputSize * pitchPercent) + topMarginSize
  x = 400#SCREEN_WIDTH/2+100
  return x, y

def inverse_lerp(a: float, b: float, value: float) -> float:
  if a == b:
    return 0.0  # Avoid division by zero
  return (value - a) / (b - a)

A4_FREQ = 440.0
A4_MIDI = 69
# MIDI note names (diatonic with accidentals)
NOTE_NAMES = ['C ', 'C#', 'D ', 'D#', 'E ', 'F ', 'F#', 'G ', 'G#', 'A ', 'A#', 'B ']
def frequency_to_note_string(freq_hz: float) -> str:
  if freq_hz <= 0:
    return "Invalid frequency"

  # Convert frequency to fractional MIDI note number
  midi = 12 * math.log2(freq_hz / A4_FREQ) + A4_MIDI
  nearest_midi = round(midi)
  cents = round((midi - nearest_midi) * 100)
  if cents > 50:
    nearest_midi += 1
    cents -= 100
  elif cents < -50:
    nearest_midi -= 1
    cents += 100

  note_name = NOTE_NAMES[nearest_midi % 12]
  octave = nearest_midi // 12 - 1  # MIDI note 0 is C-1

  sign = '+' if cents >= 0 else ''
  return f"{note_name}{octave} ({sign}{cents:0{2 + (cents < 0)}d} cents)"

def findSubIndex(preIndex: int, prevSample, nextSample, crossingPoint):
  return preIndex + (inverse_lerp(prevSample, nextSample, crossingPoint) if USE_INTERPOLATION else 1)

def getFrequency(samples, samples_per_second) -> tuple[float, float, float]:
    total = 0.0
    waveform_threshold = 0.5

    first_cross_index = -1
    third_cross_index = -1
    dist_samples = 0

    prevSample = -2
    for i, sample in enumerate(samples):
      absSample = abs(sample)
      total += absSample
      if prevSample > -2 and prevSample < waveform_threshold and sample > waveform_threshold:
        if first_cross_index < 0:
          first_cross_index = findSubIndex(i - 1, prevSample, sample, waveform_threshold)
        elif third_cross_index < 0:
          third_cross_index = findSubIndex(i - 1, prevSample, sample, waveform_threshold)
      prevSample = sample

    if third_cross_index < 0:
      frequency = 0.0
    else:
      dist_samples = (third_cross_index - first_cross_index)
      frequency = 1.0 / (dist_samples / samples_per_second)

    average_amplitude = total / len(samples)
    return average_amplitude, frequency, dist_samples

mouse_should_be_down = False
mouse_screen_x = 0
mouse_screen_y = 0
mouse_loop_exit = False

def mouseThreadLoop():
  mouse_is_down = False
  while not mouse_loop_exit:
    if mouse_should_be_down:
      mouse.move(mouse_screen_x, mouse_screen_y, absolute=True)
      mouse.press()
      mouse_is_down = True
    elif mouse_is_down:
      mouse.release()
      mouse_is_down = False
    time.sleep(0.001)
  print("Mouse thread exit")

def callback(indata, frames: int, cbTime, status):
  global mouse_should_be_down
  global mouse_screen_x
  global mouse_screen_y

  #start = time.perf_counter()
  if status:
    print(status)
    return

  if PRINT_CB_DATA:
    print(f"Time: {cbTime}, Frames: {frames}")

  if any(indata):
    mono_samples = indata[:, 0]
    average_amplitude, frequency, dist_samples = getFrequency(mono_samples, SAMPLE_FREQ)

    if average_amplitude < POWER_THRESH or frequency < LOWEST_SUPPORTED_FREQUENCY * 0.5 or frequency > MAX_FREQUENCY:
      if PRINT_NOTE:
        print(f"...{average_amplitude:.4f}")
      mouse_should_be_down = False
    else:
      pitchPercent = getPitchTrombonePercent(frequency)
      if PRINT_NOTE:
        print(f"{frequency_to_note_string(frequency)} FR:{frequency:.4f} AMP:{average_amplitude:.2f} DIST:{dist_samples:.2f} %:{(pitchPercent * 100):.1f}")
      mouse_screen_x, mouse_screen_y = getScreenPoint(pitchPercent)
      mouse_should_be_down = True

  #end = time.perf_counter()
  #print(f"TIME:{end - start:.4f}")

if MOUSE_ACTIVE:
  mouse_thread = threading.Thread(target=mouseThreadLoop)
  mouse_thread.start()

try:
  with sd.InputStream(channels=1, dtype='float32', callback=callback, blocksize=BLOCK_SIZE, samplerate=SAMPLE_FREQ, latency='low'):
    while True:
      time.sleep(0.5)
except KeyboardInterrupt:
  print("Shutdown requested")
except Exception as exc:
  print(str(exc))
print("Exiting")
mouse_loop_exit = True
