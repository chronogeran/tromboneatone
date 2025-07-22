import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
import mouse
import math

# General settings that can be changed by the user
SAMPLE_FREQ = 44100 # sample frequency in Hz
LOWEST_SUPPORTED_FREQUENCY = 200
USE_INTERPOLATION = True
POWER_THRESH = 0.1 # average amplitude cutoff for pitch detection
CONCERT_PITCH = 440 # defining a1
MIDDLE_OCTAVE = 4
PRINT_NOTE = True
MOUSE_ACTIVE = True   #if true, mouse will move to the note
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

BLOCK_LENGTH_SECONDS = 2 / LOWEST_SUPPORTED_FREQUENCY
BLOCK_SIZE = BLOCK_LENGTH_SECONDS * SAMPLE_FREQ #samples per window

def getScreenPoint(frequency):
  cFreqTable= [33,65,131,262,523,1047,2093]
  freqLowerRange = cFreqTable[MIDDLE_OCTAVE-1]
  freqMidRange = cFreqTable[MIDDLE_OCTAVE]
  freqUpperRange = cFreqTable[MIDDLE_OCTAVE+1]
  bottomMarginSize = 184
  topMarginSize = 163
  gameHeightInputSize = SCREEN_HEIGHT - bottomMarginSize - topMarginSize

  if frequency > freqMidRange:
      pitchPercent = (math.log(frequency) - math.log(freqMidRange)) / (math.log(freqUpperRange) - math.log(freqMidRange))
      pitchPercent = 1 - (pitchPercent * 0.5 + 0.5)
  else:
      pitchPercent = (math.log(frequency) - math.log(freqLowerRange)) / (math.log(freqMidRange) - math.log(freqLowerRange))
      pitchPercent = 1 - pitchPercent * 0.5
      
  """ Less Accurate
  pitchPercent = 1 - (math.log(67,frequency) - math.log(67,freqLowerRange)) / (math.log(67,freqUpperRange) - math.log(67,freqLowerRange))
  print (pitchPercent)"""
  y = int(gameHeightInputSize * pitchPercent) + topMarginSize
  x = SCREEN_WIDTH/2+100
  return x, y

def inverse_lerp(a: float, b: float, value: float) -> float:
  if a == b:
    return 0.0  # Avoid division by zero
  return (value - a) / (b - a)

# Reference: A4 = 440 Hz
A4_FREQ = 440.0
A4_MIDI = 69

# MIDI note names (diatonic with accidentals)
NOTE_NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
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
  return preIndex + USE_INTERPOLATION ? inverse_lerp(prevSample, nextSample, crossingPoint) : 1

def getFrequency(samples, samples_per_second):
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

def callback(indata, frames: int, time: CData, status):
  if status:
    print(status)
    return

  if any(indata):
    average_amplitude, frequency = getFrequency(indata, SAMPLE_FREQ)

    if average_amplitude < POWER_THRESH:
      if MOUSE_ACTIVE:
        mouse.release()
      if PRINT_NOTE:
        print(f"...{average_amplitude}")
    else:
      if MOUSE_ACTIVE:
        screenX, screenY = getScreenPoint(frequency)
        mouse.move(screenX, screenY, absolute=True)
        mouse.press()
      if PRINT_NOTE:
        print(f"{frequency_to_note_string(frequency)} {frequency} {average_amplitude} {dist_samples}")

try:
  with sd.InputStream(channels=1, dtype='float32', callback=callback, blocksize=BLOCK_SIZE, samplerate=SAMPLE_FREQ, latency='low'):
    while True:
      time.sleep(0.5)
except Exception as exc:
  print(str(exc))
