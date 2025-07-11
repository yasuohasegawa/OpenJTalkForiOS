//
//  OpenJTalkViewModel.swift
//  OpenJTalkForiOS
//
//  Created by Yasuo Hasegawa on 2025/06/27.
//

import Foundation
import AVFoundation
import AVFAudio
import CoreML

enum SynthesizerError: Error {
    case modelLoadingFailed(String)
    case phonemeMapLoadingFailed
    case preprocessingFailed(String)
    case predictionFailed(String)
    case postprocessingFailed(String)
}

class OpenJTalkViewModel: NSObject, ObservableObject {
    private let openJTalk = OpenJTalk()
    private var audioPlayer: AVAudioPlayer?
    private var engine: AVAudioEngine? = nil
    private var playerNode: AVAudioPlayerNode? = nil
    
    private var phonemeMap: [String: Int] = [:]
    private var encoderModel: MLModel! = nil
    private var decoderModel: MLModel! = nil
    private var vocoderModel: MLModel! = nil
    
    private let sampleRate = 22050.0
    
    override init(){
        super.init()
        try! loadModels()
        try! loadPhonemeMap()
        
        do {
            try saveEmptyWavFile(filename: "speech.wav")
        } catch {
            print("Error saving WAV file: \(error)")
        }
    }
    
    func loadModels() throws {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all
            
            self.encoderModel = try MLModel(contentsOf: FastSpeech2Encoder.urlOfModelInThisBundle, configuration: config)
            self.decoderModel = try MLModel(contentsOf: FastSpeech2Decoder.urlOfModelInThisBundle, configuration: config)
            
            let vocoderConfig = MLModelConfiguration()
            vocoderConfig.computeUnits = .cpuAndGPU
            self.vocoderModel = try MLModel(contentsOf: HiFiGAN.urlOfModelInThisBundle, configuration: vocoderConfig)
            print("[CoreML DEBUG] \(#function): Loaded models successfully.")
        } catch {
            throw SynthesizerError.modelLoadingFailed(error.localizedDescription)
        }
    }
    
    func loadPhonemeMap() throws {
        guard let mapURL = Bundle.main.url(forResource: "jsut_phoneme_map", withExtension: "json") else {
            throw SynthesizerError.phonemeMapLoadingFailed
        }
        
        do {
            let data = try Data(contentsOf: mapURL)
            self.phonemeMap = try JSONDecoder().decode([String: Int].self, from: data)
            print("[CoreML DEBUG] \(#function): Loaded map successfully.")
        } catch {
            throw SynthesizerError.phonemeMapLoadingFailed
        }
    }
    
    func synthesize(text: String, completion: @escaping (Result<AVAudioPCMBuffer, Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let phonemes = self.openJTalk.extractPhonemes(fromText: text)
                guard !phonemes.isEmpty else {
                    throw SynthesizerError.preprocessingFailed("Phoneme extraction returned no phonemes.")
                }
                
                // 1. Preprocess text into phoneme IDs (unchanged)
                let inputIDs = try self.preprocess(phonemes: phonemes)
                
                // 2. Run encoder to get hidden states, durations, PITCH, and ENERGY
                let encoderOutput = try self.runEncoder(inputIDs: inputIDs)
                
                // 3. Expand all features using the predicted durations
                let decoderInputs = try self.performLengthRegulation(
                    hiddenStates: encoderOutput.hs,
                    logDurations: encoderOutput.logDurations,
                    pitch: encoderOutput.pitch,
                    energy: encoderOutput.energy
                )
                
                // 4. Run the decoder with the three expanded inputs
                let melSpectrogram = try self.runDecoder(
                    expandedHiddenStates: decoderInputs.hs,
                    expandedPitch: decoderInputs.pitch,
                    expandedEnergy: decoderInputs.energy
                )
                
                // 5. Run vocoder to get waveform (unchanged)
                let waveform = try self.runVocoder(melSpectrogram: melSpectrogram)
                
                // 6. Postprocess into an audio buffer (unchanged)
                let audioBuffer = try self.postprocess(waveform: waveform)
                
                DispatchQueue.main.async {
                    completion(.success(audioBuffer))
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(error))
                }
            }
        }
    }
    
    func synthesizeAndPlayLongText(text: String, completion: @escaping (Result<Void, Error>) -> Void) {
        // 1. Split text into chunks
        let separators = CharacterSet(charactersIn: "。？！、\n")
        let chunks = text.components(separatedBy: separators).filter {
            !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        
        guard !chunks.isEmpty else {
            print("No text chunks to synthesize.")
            completion(.success(()))
            return
        }
        
        var synthesizedBuffers: [AVAudioPCMBuffer] = []
                
        // Define a recursive function to process one chunk at a time.
        func synthesizeChunk(at index: Int) {
            // Base case: If we've processed all chunks, we're done.
            guard index < chunks.count else {
                // All chunks are successfully synthesized. Now concatenate and play.
                processFinalResult(buffers: synthesizedBuffers, error: nil, completion: completion)
                return
            }
            
            let chunk = chunks[index]
            print("Synthesizing chunk \(index + 1)/\(chunks.count): '\(chunk)'")
            
            // Call the async synthesize function for the current chunk.
            synthesize(text: chunk) { [self] result in
                switch result {
                case .success(let buffer):
                    // On success, add the buffer and process the *next* chunk.
                    synthesizedBuffers.append(buffer)
                    synthesizeChunk(at: index + 1)
                    
                case .failure(let error):
                    // On failure, stop immediately and report the error.
                    print("Error on chunk \(index + 1): \(error)")
                    processFinalResult(buffers: [], error: error, completion: completion)
                }
            }
        }
        
        // Kick off the process with the first chunk (index 0).
        synthesizeChunk(at: 0)
    }

    // Helper function to handle the final result after the loop finishes or fails.
    private func processFinalResult(buffers: [AVAudioPCMBuffer], error: Error?, completion: @escaping (Result<Void, Error>) -> Void) {
        DispatchQueue.main.async {
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard !buffers.isEmpty, let finalBuffer = self.concatenate(buffers: buffers) else {
                print("Synthesis resulted in no audio buffers.")
                completion(.success(()))
                return
            }
            
            print("All chunks synthesized. Concatenating and saving audio...")
            do {
                //self.playBufferDirectly(finalBuffer)
                let url = try FileManager.default
                    .url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                    .appendingPathComponent("final_speech.wav")
                
                try self.writePCMBuffer(url: url, buffer: finalBuffer)
                self.playAudio(from: url)
                completion(.success(()))
                
            } catch let writeError {
                completion(.failure(writeError))
            }
        }
    }
    
    // This function combines audio buffers.
    func concatenate(buffers: [AVAudioPCMBuffer]) -> AVAudioPCMBuffer? {
        guard !buffers.isEmpty, let format = buffers.first?.format else { return nil }
        let totalFrameLength = buffers.reduce(0) { $0 + $1.frameLength }
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: totalFrameLength) else { return nil }
        var currentFrame: AVAudioFramePosition = 0
        for buffer in buffers {
            guard let channelData = buffer.floatChannelData?[0], let outputChannelData = outputBuffer.floatChannelData?[0] else { continue }
            let frameLength = Int(buffer.frameLength)
            memcpy(outputChannelData.advanced(by: Int(currentFrame)), channelData, frameLength * MemoryLayout<Float>.size)
            currentFrame += AVAudioFramePosition(buffer.frameLength)
        }
        outputBuffer.frameLength = totalFrameLength
        return outputBuffer
    }
    
    func writePCMBuffer(url: URL, buffer: AVAudioPCMBuffer) throws {
        let settings = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32, // or 16
            AVLinearPCMIsFloatKey: true, // or false
            AVLinearPCMIsBigEndianKey: false,
        ] as [String : Any]
        let audioFile = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try audioFile.write(from: buffer)
        print("Successfully saved audio to \(url.path)")
    }

    private func preprocess(phonemes: [String]) throws -> MLMultiArray {
        let unkID = self.phonemeMap["<unk>"] ?? 0
        let ids = phonemes.map { self.phonemeMap[$0] ?? unkID }
        let multiArray = try MLMultiArray(shape: [1, NSNumber(value: ids.count)], dataType: .int32)
        for (index, id) in ids.enumerated() {
            multiArray[index] = NSNumber(value: id)
        }
        return multiArray
    }
    
    private func runEncoder(inputIDs: MLMultiArray) throws -> (hs: MLMultiArray, logDurations: MLMultiArray, pitch: MLMultiArray, energy: MLMultiArray) {
        let encoderInput = try! MLDictionaryFeatureProvider(dictionary: ["input_ids": inputIDs])
        
        let prediction = try encoderModel.prediction(from: encoderInput)
        
        guard let hiddenStates = prediction.featureValue(for: "encoded_phonemes")?.multiArrayValue,
              let logDurations = prediction.featureValue(for: "log_durations")?.multiArrayValue,
              let pitch = prediction.featureValue(for: "pitch_predictions")?.multiArrayValue,
              let energy = prediction.featureValue(for: "energy_predictions")?.multiArrayValue else {
            throw SynthesizerError.predictionFailed("Encoder did not produce all four expected outputs.")
        }
        
        return (hiddenStates, logDurations, pitch, energy)
    }
    
    private func performLengthRegulation(hiddenStates: MLMultiArray, logDurations: MLMultiArray, pitch: MLMultiArray, energy: MLMultiArray) throws -> (hs: MLMultiArray, pitch: MLMultiArray, energy: MLMultiArray) {
        
        let logDurationsBuffer = UnsafeBufferPointer(start: logDurations.floatDataPointer, count: logDurations.count)
        
        let durations = logDurationsBuffer.map { logDurationValue in
            Int(round(max(1.0, exp(logDurationValue) - 1.0)))
        }

        let totalLength = durations.reduce(0, +)
        let hiddenSize = hiddenStates.shape[2].intValue
        
        let expandedHS = try MLMultiArray(shape: [1, NSNumber(value: totalLength), NSNumber(value: hiddenSize)], dataType: .float32)
        let expandedPitch = try MLMultiArray(shape: [1, NSNumber(value: totalLength)], dataType: .float32)
        let expandedEnergy = try MLMultiArray(shape: [1, NSNumber(value: totalLength)], dataType: .float32)
        
        let hsInPtr = hiddenStates.floatDataPointer
        let pitchInPtr = pitch.floatDataPointer
        let energyInPtr = energy.floatDataPointer
        
        let hsOutPtr = expandedHS.floatDataPointer
        let pitchOutPtr = expandedPitch.floatDataPointer
        let energyOutPtr = expandedEnergy.floatDataPointer
        
        var writeIndex = 0
        for phonemeIndex in 0..<durations.count {
            let duration = durations[phonemeIndex]
            if duration == 0 { continue }
            
            // Pointers to the single vector/value for the current phoneme
            let hsReadStartPtr = hsInPtr.advanced(by: phonemeIndex * hiddenSize)
            let pitchValue = pitchInPtr[phonemeIndex]
            let energyValue = energyInPtr[phonemeIndex]
            
            // Repeat the features for `duration` times
            for _ in 0..<duration {
                let hsWriteStartPtr = hsOutPtr.advanced(by: writeIndex * hiddenSize)
                // Copy the hidden state vector
                hsWriteStartPtr.update(from: hsReadStartPtr, count: hiddenSize)
                // Set the repeated pitch and energy values
                pitchOutPtr[writeIndex] = pitchValue
                energyOutPtr[writeIndex] = energyValue
                
                writeIndex += 1
            }
        }
        
        return (expandedHS, expandedPitch, expandedEnergy)
    }

    private func runDecoder(expandedHiddenStates: MLMultiArray, expandedPitch: MLMultiArray, expandedEnergy: MLMultiArray) throws -> MLMultiArray {
        let decoderInput = try! MLDictionaryFeatureProvider(dictionary: [
            "expanded_hidden_states": expandedHiddenStates,
            "expanded_pitch": expandedPitch,
            "expanded_energy": expandedEnergy
        ])
        
        let prediction = try decoderModel.prediction(from: decoderInput)
        
        guard let melSpectrogram = prediction.featureValue(for: "mel_spectrogram")?.multiArrayValue else {
            throw SynthesizerError.predictionFailed("Decoder did not produce mel_spectrogram.")
        }
        
        return melSpectrogram
    }
    
    private func runVocoder(melSpectrogram: MLMultiArray) throws -> MLMultiArray {
        print("melSpectrogram shape: \(melSpectrogram.shape)")
        // IMPORTANT: Transpose mel from [1, Time, 80] to [1, 80, Time] for HiFi-GAN
        let transposedMel = try self.transpose(mel: melSpectrogram)
        print("transposedMel shape: \(transposedMel.shape)")
        
        let vocoderInput = try! MLDictionaryFeatureProvider(dictionary: ["mel_spectrogram": transposedMel])
        let prediction = try vocoderModel.prediction(from: vocoderInput)
        
        guard let waveform = prediction.featureValue(for: "waveform")?.multiArrayValue else {
            throw SynthesizerError.predictionFailed("Vocoder did not produce waveform.")
        }
        
        return waveform
    }
    
    private func postprocess(waveform: MLMultiArray) throws -> AVAudioPCMBuffer {
        
        let sampleCount = waveform.count
        let waveformPtr = waveform.floatDataPointer
        
        // Create a new buffer to hold the verified data.
        guard let verifiedWaveform = malloc(sampleCount * MemoryLayout<Float32>.size)?.bindMemory(to: Float32.self, capacity: sampleCount) else {
            throw SynthesizerError.postprocessingFailed("Failed to allocate memory for waveform.")
        }
        defer {
            // Make sure we free the memory we allocated.
            free(verifiedWaveform)
        }

        // Clamp values to the [-1.0, 1.0] range to prevent clipping/distortion.
        for i in 0..<sampleCount {
            let sample = waveformPtr[i]
            // This is a very important step!
            verifiedWaveform[i] = max(-1.0, min(1.0, sample))
        }
        
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: self.sampleRate, channels: 1, interleaved: false) else {
            throw SynthesizerError.postprocessingFailed("Failed to create AVAudioFormat.")
        }
        
        let frameLength = AVAudioFrameCount(waveform.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameLength) else {
            throw SynthesizerError.postprocessingFailed("Failed to create AVAudioPCMBuffer.")
        }
        
        buffer.frameLength = frameLength
        
        // Copy the waveform data from the MLMultiArray to the audio buffer
        if let channelData = buffer.floatChannelData?[0] {
            channelData.update(from: verifiedWaveform, count: sampleCount)
        }
        
        return buffer
    }

    // Transposes an MLMultiArray from [1, Time, Channels] to [1, Channels, Time].
    private func transpose(mel: MLMultiArray) throws -> MLMultiArray {
        let timeSteps = mel.shape[1].intValue
        let numChannels = mel.shape[2].intValue // Should be 80
        
        let transposedMel = try MLMultiArray(shape: [1, NSNumber(value: numChannels), NSNumber(value: timeSteps)], dataType: .float32)
        
        let inputPtr = mel.floatDataPointer
        let outputPtr = transposedMel.floatDataPointer
        
        for t in 0..<timeSteps {
            for c in 0..<numChannels {
                let inputIndex = t * numChannels + c
                let outputIndex = c * timeSteps + t
                outputPtr[outputIndex] = inputPtr[inputIndex]
            }
        }
        return transposedMel
    }
    
    func speakWithOpenJTalk(text: String){
        let tempURL = try? FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true).appendingPathComponent("speech.wav")
        
        print(">>>>> speakWithOpenJTalk called")
        openJTalk.synthesize(text, pitch:1.0, to: tempURL!) { [weak self] error in
            guard let self = self else { return }
            print(">>>>> speakWithOpenJTalk synthesize completed")
            // Always update UI on the main thread
            DispatchQueue.main.async {

                if let error = error {
                    print("Synthesis failed with error: \(error.localizedDescription)")
                    // Optionally, show an alert to the user
                    return
                }

                Task{
                    try await Task.sleep(nanoseconds: 1_000_000_000)
                    print("Synthesis successful. Playing audio from: \(tempURL!.path)")
                    DispatchQueue.main.async {
                        self.playAudio(from: tempURL!)
                    }
                }
            }
        }
    }
    
    func extractPhoneme(text: String) -> String{
        print(">>>>> extractPhoneme called")
        let phonemes = openJTalk.extractPhonemes(fromText: text)
        print(">>>>> extractPhoneme completed: \(phonemes)")
        return phonemes.joined(separator: ",")
    }
    
    private func playAudio(from url: URL) {
        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            self.audioPlayer = try AVAudioPlayer(contentsOf: url)
            self.audioPlayer?.play()

        } catch {
            print("AVAudioPlayer failed with error: \(error.localizedDescription)")
        }
    }
    
    func playBufferDirectly(_ buffer: AVAudioPCMBuffer) {
        engine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()

        guard let engine = engine, let playerNode = playerNode else {
            print("❌ Failed to set up AVAudioEngine or PlayerNode")
            return
        }

        engine.attach(playerNode)
        engine.connect(playerNode, to: engine.mainMixerNode, format: buffer.format)

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)
            
            try engine.start()
            playerNode.scheduleBuffer(buffer, at: nil, options: .interrupts) {
                print("✅ Playback finished")
            }
            playerNode.play()
            print("▶️ Playing directly from AVAudioPCMBuffer")
        } catch {
            print("❌ Error starting audio engine or playback: \(error.localizedDescription)")
        }
    }
    
    func saveEmptyWavFile(filename: String, sampleRate: Int = 44100, channels: Int = 1, bitsPerSample: Int = 16) throws {
        let fileManager = FileManager.default
        let url = try fileManager
            .url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent(filename)

        // No audio data, just the header
        let byteRate = sampleRate * channels * bitsPerSample / 8
        let blockAlign = channels * bitsPerSample / 8
        let dataChunkSize = 0
        let fmtChunkSize = 16
        let audioFormat: UInt16 = 1 // PCM

        var wavHeader = Data()

        // RIFF header
        wavHeader.append("RIFF".data(using: .ascii)!)                     // ChunkID
        wavHeader.append(UInt32(36 + dataChunkSize).littleEndianData)     // ChunkSize = 36 + data size
        wavHeader.append("WAVE".data(using: .ascii)!)                     // Format

        // fmt subchunk
        wavHeader.append("fmt ".data(using: .ascii)!)                     // Subchunk1ID
        wavHeader.append(UInt32(fmtChunkSize).littleEndianData)           // Subchunk1Size
        wavHeader.append(UInt16(audioFormat).littleEndianData)            // AudioFormat (1 = PCM)
        wavHeader.append(UInt16(channels).littleEndianData)               // NumChannels
        wavHeader.append(UInt32(sampleRate).littleEndianData)             // SampleRate
        wavHeader.append(UInt32(byteRate).littleEndianData)               // ByteRate
        wavHeader.append(UInt16(blockAlign).littleEndianData)             // BlockAlign
        wavHeader.append(UInt16(bitsPerSample).littleEndianData)          // BitsPerSample

        // data subchunk
        wavHeader.append("data".data(using: .ascii)!)                     // Subchunk2ID
        wavHeader.append(UInt32(dataChunkSize).littleEndianData)          // Subchunk2Size

        try wavHeader.write(to: url)
        print("Empty WAV saved to: \(url.path)")
    }

    func debugSave(multiArray: MLMultiArray, filename: String) {
        do {
            let url = try FileManager.default
                .url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                .appendingPathComponent(filename)

            let pointer = multiArray.floatDataPointer
            let count = multiArray.count
            
            var text = "Shape: \(multiArray.shape), Count: \(count)\n"
            text += "First 100 values:\n"
            
            // Print the first 100 values to see their range
            for i in 0..<min(100, count) {
                text += "\(pointer[i]), "
            }
            
            text += "\n\nFull Data:\n"
            // If it's not too big, write more data
            if count < 5000 {
                for i in 0..<count {
                    text += "\(pointer[i]), "
                    if (i + 1) % 10 == 0 { // Newline every 10 numbers
                        text += "\n"
                    }
                }
            } else {
                text += "Data too large to write fully."
            }

            try text.write(to: url, atomically: true, encoding: .utf8)
            print("✅ [DEBUG] Saved data to \(filename)")

        } catch {
            print("❌ [DEBUG] Failed to save debug file \(filename): \(error)")
        }
    }
    
}
