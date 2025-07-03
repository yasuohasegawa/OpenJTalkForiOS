# OpenJTalk iOS Integration Guide

This project integrates **OpenJTalk**, a Japanese Text-to-Speech (TTS) engine, for iOS development.
**Note:** OpenJTalk supports **only Japanese** language TTS.

---

## 📦 Setup Instructions

### 1. Build Settings Configuration

#### Header Search Paths

Add the following to **Build Settings > Header Search Paths**:

```
OpenJTalk/Headers/include
```

* Set this entry to **recursive**.

#### Library Search Paths

Add the following to **Build Settings > Library Search Paths**:

```
OpenJTalk/Libraries
```

* Also set to **recursive**.

#### Bridging Header (for Swift)

Set the **Objective-C Bridging Header** path to:

```
<your path>/SpeechRecognizerTest-Bridging-Header.h
```

---

### 2. Copy Dictionary Files (Required for OpenJTalk)

In **Build Phases**, do the following:

1. Click the **+** button in the top-left corner and select **"New Copy Files Phase"**.
2. Set **Destination** to `Resources`.
3. Click the **+** button and choose **Add Other**.
4. Select the `dic` folder.

> ⚠️ Important: **Do not** add the `dic/` folder to the "Copy Bundle Resources" section.
> Doing so will prevent the dictionary from loading properly at runtime.

---

### 3. Testing Custom Voices

To test with other voices, copy your `.htsvoice` files into the following folder:

```
OpenJTalk/Resources/
```

---

## 🚫 iPhone Simulator Limitation

This project does **not** work on the iPhone Simulator. Please test on a real device.

---

## 🔗 Resources Used to Build OpenJTalk for iOS

We compiled OpenJTalk for iOS using the following sources:

* [OpenJTalk 1.11](https://sourceforge.net/projects/open-jtalk/files/Open%20JTalk/open_jtalk-1.11/open_jtalk-1.11.tar.gz/download)
* [HTS Engine API 1.10](https://sourceforge.net/projects/hts-engine/files/hts_engine%20API/hts_engine_API-1.10/hts_engine_API-1.10.tar.gz/download)
* [OpenJTalk Dictionary (UTF-8) 1.11](https://sourceforge.net/projects/open-jtalk/files/Dictionary/open_jtalk_dic-1.11/open_jtalk_dic_utf_8-1.11.tar.gz/download)

A helper shell script `build-ios.sh` is included for convenience.
