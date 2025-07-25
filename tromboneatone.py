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

# General settings that can be changed by the user
SAMPLE_FREQ = 44100 # sample frequency in Hz
LOWEST_SUPPORTED_FREQUENCY = 200
USE_INTERPOLATION = True
POWER_THRESH = 0.08 # average amplitude cutoff for pitch detection
MIDDLE_OCTAVE = 4
PRINT_NOTE = True
PRINT_CB_DATA = False
MOUSE_ACTIVE = True   #if true, mouse will move to the note
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

BLOCK_LENGTH_SECONDS = 2 / LOWEST_SUPPORTED_FREQUENCY
BLOCK_SIZE = int(BLOCK_LENGTH_SECONDS * SAMPLE_FREQ) #samples per window
MAX_FREQUENCY = 4500

cFreqTable= [33,65,131,262,523,1047,2093]
logFreqLowerRange = math.log(cFreqTable[MIDDLE_OCTAVE-1])
freqMidRange = cFreqTable[MIDDLE_OCTAVE]
logFreqMidRange = math.log(freqMidRange)
logFreqUpperRange = math.log(cFreqTable[MIDDLE_OCTAVE+1])
# 0 = high, 1 = low
def getPitchTrombonePercent(frequency) -> float:
  if frequency > freqMidRange:
    pitchPercent = (math.log(frequency) - logFreqMidRange) / (logFreqUpperRange - logFreqMidRange)
    pitchPercent = 1 - (pitchPercent * 0.5 + 0.5)
  else:
    pitchPercent = (math.log(frequency) - logFreqLowerRange) / (logFreqMidRange - logFreqLowerRange)
    pitchPercent = 1 - pitchPercent * 0.5
  return np.clip(pitchPercent, 0, 1)

def getScreenPoint(pitchPercent):
  bottomMarginSize = 184 # TODO proportional to resolution
  topMarginSize = 163
  gameHeightInputSize = SCREEN_HEIGHT - bottomMarginSize - topMarginSize
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
  return f"{note_name}{octave} ({sign}{cents} cents)"

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

mouseDown = False
screenX = 0
screenY = 0
mouseDone = False

def mouseThreadLoop():
  mouseIsDown = False
  while not mouseDone:
    if mouseDown:
      mouse.move(screenX, screenY, absolute=True)
      mouse.press()
      mouseIsDown = True
    elif mouseIsDown:
      mouse.release()
      mouseIsDown = False
    time.sleep(0.001)
  print("Mouse done!")

def callback(indata, frames: int, cbTime, status):
  global mouseDown
  global screenX
  global screenY

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
      mouseDown = False
    else:
      pitchPercent = getPitchTrombonePercent(frequency)
      if PRINT_NOTE:
        print(f"{frequency_to_note_string(frequency)} FR:{frequency:.4f} AMP:{average_amplitude:.2f} DIST:{dist_samples:.2f} %:{pitchPercent:.2f}")
      screenX, screenY = getScreenPoint(pitchPercent)
      mouseDown = True

  #end = time.perf_counter()
  #print(f"TIME:{end - start:.4f}")

if MOUSE_ACTIVE:
  mouseTread = threading.Thread(target=mouseThreadLoop)
  mouseTread.start()

try:
  with sd.InputStream(channels=1, dtype='float32', callback=callback, blocksize=BLOCK_SIZE, samplerate=SAMPLE_FREQ, latency='low'):
    while True:
      time.sleep(0.5)
except KeyboardInterrupt:
  print("Shutdown requested")
except Exception as exc:
  print(str(exc))
print("Exiting")
mouseDone = True
