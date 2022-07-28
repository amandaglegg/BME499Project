//
//  ECGDocument.swift
//  GoodHealth
//
//  Created by Derrick Ushko on 2022-07-15.
//

import SwiftUI
import UniformTypeIdentifiers

struct ECGDocument: FileDocument {
    
    static var readableContentTypes: [UTType] { [.plainText] }

    var sample: String

    init(sample: String) {
        self.sample = sample
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents,
              let string = String(data: data, encoding: .utf8)
        else {
            throw CocoaError(.fileReadCorruptFile)
        }
        sample = string
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        return FileWrapper(regularFileWithContents: sample.data(using: .utf8)!)
    }
    
}
