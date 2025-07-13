//
//  G2PConverter.swift
//  OpenJTalkForiOS
//
//  Created by Yasuo Hasegawa on 2025/07/13.
//

import Foundation

class G2PConverter {
    private let cmuDict: CMUDict

    init() {
        do {
            self.cmuDict = try CMUDict()
        } catch {
            fatalError("Could not initialize G2PConverter: \(error)")
        }
    }

    func convert(text: String) -> [String] {
        print("--- RUNNING THE DEFINITIVE G2P CONVERTER ---")
        let cleanedText = text.uppercased()
        let tokens = cleanedText.matches(for: "[A-Z']+|[^A-Z'\\s]")
        var finalPhonemes: [String] = []

        for token in tokens {
           if let phonemes = cmuDict.getPhonemes(for: token) {
               finalPhonemes.append(contentsOf: phonemes)
           } else if token.allSatisfy({ $0.isLetter }) {
               print("⚠️ Word not found in CMUdict: '\(token)'. Spelling it out.")
               finalPhonemes.append(contentsOf: token.map { String($0) })
           } else {
               finalPhonemes.append(token)
           }
        }
        return finalPhonemes
   }
}
