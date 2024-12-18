<img src="tools/UI/icons/sample_tools_64.png" alt="Description" style="float: left; margin-right: 16px;">

# SampleTools

A collection of tools to assist in assembling sampled instruments with a more specific focus on Decent Sampler

## Installation

This project was developed and tested with Python 3.12.7

Additional required packages all listed in provided requirements.txt

- chunkmuncher
- librosa
- matplotlib
- mutagen
- noisereduce
- numpy
- pandas
- pillow
- PyQt5
- PyQt5_sip
- QDarkStyle
- scipy
- sounddevic
- soundfile
- webcolors
- pyrubberband (only for upsample)

  (required rubberband executable available here https://breakfastquay.com/rubberband/)

## Usage

Execute sample_tools_UI.pyw

### Common features for most tools

- Drag and drop of files or directories (only the first level is scanned)
- Context menus with right click on many widgets for examples or presets
- Regular or looped Sample playback (Double click or Space Bar)
- Keyboard shortcuts for file list (ctrl+A, ctrl+I, Delete)
- Visualization of waveform and loop point (W, L)
- Help is provided using tooltips for most widgets
- Backup, increment to avoid overwriting

FLAC format stores metadata using custom ID3 tags and is recommended to get the most features



<img src="tools/UI/icons/smp2ds_64.png" alt="Description" style="float: left; margin-right: 16px;">

## SMP2Ds

Create Decent Sampler presets from samples in wav or flac format (limited support of aif)
Drag and drop a directory with a 'Samples' subdirectory in it to generate a dspreset file

- Samples are added and set according to their 'smpl' chunk/metadata and file name pattern matching
  respect for pitch, pitch fraction, loop points
- Fake release, fake legato and fake robin features
- Automatic creation of a working UI with a customisable color theme
- wav, flac and aif files are supported for the convolution reverb, they must be located in an IR subdirectory
- dslibrary file generation from a dspreset directory ready to use or distribute

### smp_attrib_cfg.json

Sample attributes Config

List of sample attributes offered by Decent Sampler and *hopefully* supported by this tool

Attribute values can be provided in the sample name (with some limitations depending on how the name is formatted)
or using ID3 tags (flac only at the moment, it should support any attribute, but I didn't test everything...)

This file found in this directory is required by this tool and should only be modified to recognize future Decent
Sampler features

- **Attribute names are case-sensitive!**

- 'smp_attrib' key defines a list of basic attributes used by this tool
- 'ds_smp_attrib' key defines a list of specific attributes recognized by Decent Sampler and described in its
  documentation
- 'num_attrib' key defines a list of attributes which should be considered as numeric values

### instr_range.json

This file is used to define custom instrument ranges used by this tool when pressing the **limit** button

The key name is the name of the instrument with min and max MIDI note number provided as a list


<img src="tools/UI/icons/split_tool_64.png" alt="Description" style="float: left; margin-right: 16px;">

## Split Audio Tool

Split and trim audio file(s) by detecting silences
typical usage : note by note instrument recordings, audio sample CDs tracks

Input audio files are split into several samples and renamed according user defined options

- number increment
- custom suffixes
- note number or name using automatic pitch detection

- Automatic micro fade in/out to eliminate potential popping

<img src="tools/UI/icons/rename_tool_64.png" alt="Description" style="float: left; margin-right: 16px;">

## Rename Sample Tool

Renaming, conversion of audio files using pattern matching, pitch detection

Update their 'smpl' chunk/metadata accordingly, so they are properly conformed by SMP2ds or Kontakt (wav only)
wav and flac output

<img src="tools/UI/icons/loop_tool_64.png" alt="Description" style="float: left; margin-right: 16px;">

## Loop Tool

Detect loop points or modify audio files to make them loop

- Batch auto-detection of loop points using zero crossing and auto correlation
- FFT re-synthesis

<img src="tools/UI/icons/st_tool_64.png" alt="Description" style="float: left; margin-right: 16px;">

## Stereo Tool

Apply pseudo-stereo/stereo imaging effect to mono audio file(s)

- Haas, Velvet, Convolution, Side Convolution

<img src="tools/UI/icons/upsample_tool_64.png" alt="Description" style="float: left; margin-right: 16px;">

## Upsample Tool

Up-sample audio file(s) using spectral band replication and denoising to improve old 8 bits samples

## License

MIT License

Copyright (c) 2024 Michel 'Mitch' Pecqueur

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.