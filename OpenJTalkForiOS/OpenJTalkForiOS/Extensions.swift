//
//  Extensions.swift
//  SpeechRecognizerTest
//
//  Created by Yasuo Hasegawa on 2025/06/24.
//

import Foundation
import AVFoundation
import Speech

extension AVSpeechSynthesizer {
    func write(_ utterance: AVSpeechUtterance) -> AsyncStream<AVAudioBuffer> {
        AsyncStream(AVAudioBuffer.self) { continuation in
            write(utterance) { (buffer: AVAudioBuffer) in
                if buffer.audioBufferList.pointee.mBuffers.mDataByteSize > 0 {
                    continuation.yield(buffer)
                } else {
                    continuation.finish()
                }
            }
        }
    }
}

extension FixedWidthInteger {
    var littleEndianData: Data {
        withUnsafeBytes(of: self.littleEndian) { Data($0) }
    }
}
