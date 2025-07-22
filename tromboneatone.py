'''
Guitar tuner script based on the Harmonic Product Spectrum (HPS)

MIT License
Copyright (c) 2021 chciken

Adapted into tromboneatone, a program to use your otamatone as a controller for Trombone Champ by Mcall
'''

import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
import mouse
import math

# General settings that can be changed by the user
SAMPLE_FREQ = 20000 # sample frequency in Hz
BLOCK_SIZE = 300 #samples per window
POWER_THRESH = 0.1 # average amplitude cutoff for pitch detection
CONCERT_PITCH = 440 # defining a1
PRINT_NOTE = True
mouseActive = True   #if true, mouse will move to the note
screenWidth = 1920
screenHeight = 1080

ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]
def find_closest_note(pitch):
  """
  This function finds the closest note for a given pitch
  Parameters:
    pitch (float): pitch given in hertz
  Returns:
    closest_note (str): e.g. a, g#, ..
    closest_pitch (float): pitch of the closest note in hertz
  """
  i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
  closest_note = ALL_NOTES[i%12] + str(4 + (i + 9) // 12)
  closest_pitch = CONCERT_PITCH*2**(i/12)
  return closest_note, closest_pitch

def getScreenPoint(frequency):
  cFreqTable= [33,65,131,262,523,1047,2093]
  middleOctive = 4
  freqLowerRange = cFreqTable[middleOctive-1]
  freqMidRange = cFreqTable[middleOctive]
  freqUpperRange = cFreqTable[middleOctive+1]
  bottomMarginSize = 135
  bottomMarginSizeC = 184
  topMarginSize = 140
  topMarginSizeC = 163
  gameHeightInputSizeC = screenHeight - bottomMarginSizeC - topMarginSizeC
  gameHeightInputSize = screenHeight - bottomMarginSize - topMarginSize

  if frequency > freqMidRange:
      pitchPercent = (math.log(frequency) - math.log(freqMidRange)) / (math.log(freqUpperRange) - math.log(freqMidRange))
      pitchPercent = 1 - (pitchPercent * 0.5 + 0.5)
  else:
      pitchPercent = (math.log(frequency) - math.log(freqLowerRange)) / (math.log(freqMidRange) - math.log(freqLowerRange))
      pitchPercent = 1 - pitchPercent * 0.5
      
  """ Less Accurate
  pitchPercent = (math.log(67,frequency) - math.log(67,freqLowerRange)) / (math.log(67,freqUpperRange) - math.log(67,freqLowerRange))
  pitchPercent = 1 - pitchPercent
  print (pitchPercent)"""
  y = int(gameHeightInputSizeC * pitchPercent) + topMarginSizeC
  x = screenWidth/2+100
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

def findSubIndex(preIndex, prevSample, nextSample, crossingPoint):
  return preIndex + 1#preIndex + inverse_lerp(prevSample, nextSample, crossingPoint)

def getFrequency(samples, samples_per_second):
    total = 0.0
    negative = False
    waveform_threshold = 0.5

    first_cross_index = -1
    second_cross_index = -1
    third_cross_index = -1

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
        frequency = 0.0  # Not enough zero crossings to estimate frequency
    else:
        frequency = 1.0 / ((third_cross_index - first_cross_index) / samples_per_second)

    average_amplitude = total / len(samples)
    return average_amplitude, frequency

def callback(indata, frames, time, status):
  if status:
    print(status)
    return

  if any(indata):
    average_amplitude, max_freq = getFrequency(indata, SAMPLE_FREQ)

    if average_amplitude < POWER_THRESH:
      if mouseActive:
        mouse.release()
      if PRINT_NOTE:
        print(f"...{average_amplitude}")
    else:
      if mouseActive:
        screenX, screenY = getScreenPoint(max_freq)
        mouse.move(screenX, screenY, absolute=True)
        mouse.press()
      if PRINT_NOTE:
        print(f"{frequency_to_note_string(max_freq)} {max_freq} {average_amplitude}")

try:
  with sd.InputStream(channels=1, callback=callback, blocksize=BLOCK_SIZE, samplerate=SAMPLE_FREQ):
    while True:
      time.sleep(0.5)
except Exception as exc:
  print(str(exc))
