//
//  MNISTMetalRunner.swift
//  TaichiJitExampleApp
//
//  Created by Codex on 2026/02/11.
//

import Foundation
import Metal

enum MNISTRunnerError: LocalizedError {
    case metalUnavailable
    case commandQueueUnavailable
    case missingBundleResource(String)
    case invalidDatasetFormat(String)
    case invalidMetadata(String)
    case missingKernel(String)
    case missingField(String)
    case commandBufferFailed(String)

    var errorDescription: String? {
        switch self {
        case .metalUnavailable:
            return "Metal device is unavailable."
        case .commandQueueUnavailable:
            return "Failed to create Metal command queue."
        case .missingBundleResource(let name):
            return "Missing app resource: \(name)"
        case .invalidDatasetFormat(let detail):
            return "Invalid MNIST dataset format: \(detail)"
        case .invalidMetadata(let detail):
            return "Invalid Taichi metadata: \(detail)"
        case .missingKernel(let name):
            return "Kernel not found in metadata: \(name)"
        case .missingField(let name):
            return "Field offset not found in metadata: \(name)"
        case .commandBufferFailed(let detail):
            return "Compute command failed: \(detail)"
        }
    }
}

struct MNISTTrainingProgress {
    let epoch: Int
    let sample: Int
    let totalSamples: Int
    let averageLoss: Float
    let accuracy: Float
}

struct MNISTTrainingSummary {
    let epochs: Int
    let trainLoss: Float
    let trainAccuracy: Float
    let testAccuracy: Float
    let stoppedByLossThreshold: Bool
}

struct MNISTInferenceResult {
    let image: [Float]
    let label: Int
    let predictedLabel: Int
    let confidence: Float
}

final class MNISTMetalRunner {
    private struct Args1 {
        var value: Float
        var pad0: UInt32 = 0
        var pad1: UInt32 = 0
        var pad2: UInt32 = 0
    }

    private struct MNISTBinaryDataset {
        let imageSize: Int
        let trainCount: Int
        let testCount: Int
        let trainImages: [Float]
        let trainLabels: [UInt8]
        let testImages: [Float]
        let testLabels: [UInt8]

        static func load(from url: URL) throws -> MNISTBinaryDataset {
            let data = try Data(contentsOf: url)
            let headerSize = 4 + 4 + 4 + 4 + 4
            guard data.count >= headerSize else {
                throw MNISTRunnerError.invalidDatasetFormat("file too small")
            }

            var offset = 0
            let magic = String(data: data[offset..<offset + 4], encoding: .ascii) ?? ""
            offset += 4
            guard magic == "MNST" else {
                throw MNISTRunnerError.invalidDatasetFormat("bad magic: \(magic)")
            }

            let version = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            guard version == 1 else {
                throw MNISTRunnerError.invalidDatasetFormat("unsupported version: \(version)")
            }

            let imageSizeU32 = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            let trainCountU32 = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            let testCountU32 = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4

            let imageSize = Int(imageSizeU32)
            let trainCount = Int(trainCountU32)
            let testCount = Int(testCountU32)

            let trainImageBytes = trainCount * imageSize * MemoryLayout<Float>.size
            let trainLabelBytes = trainCount
            let testImageBytes = testCount * imageSize * MemoryLayout<Float>.size
            let testLabelBytes = testCount
            let totalBytes = headerSize + trainImageBytes + trainLabelBytes + testImageBytes + testLabelBytes
            guard data.count == totalBytes else {
                throw MNISTRunnerError.invalidDatasetFormat("byte size mismatch")
            }

            let trainImages: [Float] = data[(offset)..<(offset + trainImageBytes)].withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            offset += trainImageBytes

            let trainLabels: [UInt8] = Array(data[offset..<(offset + trainLabelBytes)])
            offset += trainLabelBytes

            let testImages: [Float] = data[(offset)..<(offset + testImageBytes)].withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            offset += testImageBytes

            let testLabels: [UInt8] = Array(data[offset..<(offset + testLabelBytes)])

            return MNISTBinaryDataset(
                imageSize: imageSize,
                trainCount: trainCount,
                testCount: testCount,
                trainImages: trainImages,
                trainLabels: trainLabels,
                testImages: testImages,
                testLabels: testLabels
            )
        }
    }

    private struct AOTMetadata: Decodable {
        struct Field: Decodable {
            let field_name: String
            let mem_offset_in_parent: Int
        }

        struct Kernel: Decodable {
            struct Task: Decodable {
                struct BufferBind: Decodable {
                    struct BufferDesc: Decodable {
                        let type: Int
                    }

                    let buffer: BufferDesc
                }

                let name: String
                let advisory_num_threads_per_group: Int
                let advisory_total_num_threads: Int
                let buffer_binds: [BufferBind]
            }

            let name: String
            let tasks_attribs: [Task]
        }

        let fields: [Field]
        let kernels: [Kernel]
        let root_buffer_size: Int
    }

    private struct TaskPipeline {
        let pipeline: MTLComputePipelineState
        let threadCount: Int
        let threadsPerGroup: Int
        let requiresArgs: Bool
    }

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let rootBuffer: MTLBuffer
    private let dataset: MNISTBinaryDataset
    private let fieldOffsets: [String: Int]
    private let kernelTasks: [String: [TaskPipeline]]

    private let xOffset: Int
    private let labelOffset: Int
    private let logitsOffset: Int
    private let lossOffset: Int
    private let numClasses = 10

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MNISTRunnerError.metalUnavailable
        }
        guard let commandQueue = device.makeCommandQueue() else {
            throw MNISTRunnerError.commandQueueUnavailable
        }
        self.device = device
        self.commandQueue = commandQueue

        let bundle = Bundle.main

        guard let metadataURL = bundle.url(forResource: "mnist_metadata", withExtension: "json"
//                                           , subdirectory: "MNIST"
        ) else {
            throw MNISTRunnerError.missingBundleResource("MNIST/mnist_metadata.json")
        }
        let metadataData = try Data(contentsOf: metadataURL)
        let metadata = try JSONDecoder().decode(AOTMetadata.self, from: metadataData)

        guard let datasetURL = bundle.url(forResource: "mnist_subset", withExtension: "bin"
//                                          , subdirectory: "MNIST"
        ) else {
            throw MNISTRunnerError.missingBundleResource("MNIST/mnist_subset.bin")
        }
        self.dataset = try MNISTBinaryDataset.load(from: datasetURL)
        guard dataset.imageSize == 28 * 28 else {
            throw MNISTRunnerError.invalidDatasetFormat("unexpected image size: \(dataset.imageSize)")
        }

        guard let rootBuffer = device.makeBuffer(length: metadata.root_buffer_size, options: .storageModeShared) else {
            throw MNISTRunnerError.invalidMetadata("failed to allocate root buffer")
        }
        self.rootBuffer = rootBuffer
        rootBuffer.contents().initializeMemory(as: UInt8.self, repeating: 0, count: metadata.root_buffer_size)

        var offsets: [String: Int] = [:]
        for field in metadata.fields {
            offsets[field.field_name] = field.mem_offset_in_parent
        }
        self.fieldOffsets = offsets
        self.xOffset = try Self.requireFieldOffset("x", from: offsets)
        self.labelOffset = try Self.requireFieldOffset("label", from: offsets)
        self.logitsOffset = try Self.requireFieldOffset("logits", from: offsets)
        self.lossOffset = try Self.requireFieldOffset("loss", from: offsets)

        var plans: [String: [TaskPipeline]] = [:]
        for kernel in metadata.kernels {
            var tasks: [TaskPipeline] = []
            for task in kernel.tasks_attribs {
                guard let libURL = bundle.url(
                    forResource: task.name,
                    withExtension: "metallib"
//                    ,subdirectory: "Shaders/MNIST"
                ) else {
                    throw MNISTRunnerError.missingBundleResource("Shaders/MNIST/\(task.name).metallib")
                }
                let library = try device.makeLibrary(URL: libURL)
                guard let function = library.makeFunction(name: "main0") else {
                    throw MNISTRunnerError.invalidMetadata("missing main0 in \(task.name)")
                }
                let pipeline = try device.makeComputePipelineState(function: function)
                let requiresArgs = task.buffer_binds.contains { $0.buffer.type == 2 }
                tasks.append(
                    TaskPipeline(
                        pipeline: pipeline,
                        threadCount: max(task.advisory_total_num_threads, 1),
                        threadsPerGroup: max(task.advisory_num_threads_per_group, 1),
                        requiresArgs: requiresArgs
                    )
                )
            }
            plans[kernel.name] = tasks
        }
        self.kernelTasks = plans

        _ = try requireKernel("mnist_init_params")
        _ = try requireKernel("mnist_forward")
        _ = try requireKernel("mnist_backward")
        _ = try requireKernel("mnist_apply_grad")
    }

    func train(
        epochs: Int,
        learningRate: Float,
        lossThreshold: Float?,
        progress: ((MNISTTrainingProgress) -> Void)?
    ) throws -> MNISTTrainingSummary {
        try runKernel("mnist_init_params", argsValue: nil)

        let targetEpochs = max(1, epochs)
        let trainCount = dataset.trainCount
        var lastAvgLoss: Float = 0
        var lastAcc: Float = 0
        var completedEpochs = 0
        var stoppedByLossThreshold = false

        for epoch in 0..<targetEpochs {
            var lossSum: Float = 0
            var correct = 0

            for i in 0..<trainCount {
                writeTrainSample(index: i)
                try runTrainStep(learningRate: learningRate)

                let loss = readFloat(offset: lossOffset)
                lossSum += loss
                let prediction = argmaxLogits()
                if prediction == Int(dataset.trainLabels[i]) {
                    correct += 1
                }

                if i % 100 == 0 || i + 1 == trainCount {
                    let avgLoss = lossSum / Float(i + 1)
                    let acc = Float(correct) / Float(i + 1)
                    progress?(
                        MNISTTrainingProgress(
                            epoch: epoch + 1,
                            sample: i + 1,
                            totalSamples: trainCount,
                            averageLoss: avgLoss,
                            accuracy: acc
                        )
                    )
                    lastAvgLoss = avgLoss
                    lastAcc = acc
                    if let threshold = lossThreshold, avgLoss <= threshold {
                        stoppedByLossThreshold = true
                        break
                    }
                }
            }
            completedEpochs = epoch + 1
            if stoppedByLossThreshold {
                break
            }
        }

        let testAccuracy = try evaluateTestAccuracy(sampleLimit: min(dataset.testCount, 300))
        return MNISTTrainingSummary(
            epochs: completedEpochs,
            trainLoss: lastAvgLoss,
            trainAccuracy: lastAcc,
            testAccuracy: testAccuracy,
            stoppedByLossThreshold: stoppedByLossThreshold
        )
    }

    func inferRandomTestSample() throws -> MNISTInferenceResult {
        let index = Int.random(in: 0..<dataset.testCount)
        return try inferTestSample(index: index)
    }

    private func inferTestSample(index: Int) throws -> MNISTInferenceResult {
        writeTestSample(index: index)
        try runKernel("mnist_forward", argsValue: nil)
        let logits = readLogits()
        let predicted = argmax(logits)
        let confidence = softmaxConfidence(logits: logits, index: predicted)

        let imageStart = index * dataset.imageSize
        let imageEnd = imageStart + dataset.imageSize
        let image = Array(dataset.testImages[imageStart..<imageEnd])
        let label = Int(dataset.testLabels[index])
        return MNISTInferenceResult(
            image: image,
            label: label,
            predictedLabel: predicted,
            confidence: confidence
        )
    }

    private func evaluateTestAccuracy(sampleLimit: Int) throws -> Float {
        var correct = 0
        for i in 0..<sampleLimit {
            writeTestSample(index: i)
            try runKernel("mnist_forward", argsValue: nil)
            let pred = argmaxLogits()
            if pred == Int(dataset.testLabels[i]) {
                correct += 1
            }
        }
        return Float(correct) / Float(sampleLimit)
    }

    private func runTrainStep(learningRate: Float) throws {
        guard let argsBuffer = makeArgsBuffer(value: learningRate) else {
            throw MNISTRunnerError.invalidMetadata("failed to allocate args buffer")
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MNISTRunnerError.commandBufferFailed("failed to create command buffer")
        }
        try encodeKernel("mnist_forward", into: commandBuffer, argsBuffer: nil)
        try encodeKernel("mnist_backward", into: commandBuffer, argsBuffer: nil)
        try encodeKernel("mnist_apply_grad", into: commandBuffer, argsBuffer: argsBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MNISTRunnerError.commandBufferFailed(error.localizedDescription)
        }
    }

    private func runKernel(_ name: String, argsValue: Float?) throws {
        let argsBuffer = argsValue == nil ? nil : makeArgsBuffer(value: argsValue!)
        if argsValue != nil && argsBuffer == nil {
            throw MNISTRunnerError.invalidMetadata("failed to allocate args buffer")
        }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            throw MNISTRunnerError.commandBufferFailed("failed to create command buffer")
        }
        try encodeKernel(name, into: commandBuffer, argsBuffer: argsBuffer)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw MNISTRunnerError.commandBufferFailed(error.localizedDescription)
        }
    }

    private func encodeKernel(
        _ name: String,
        into commandBuffer: MTLCommandBuffer,
        argsBuffer: MTLBuffer?
    ) throws {
        let tasks = try requireKernel(name)
        for task in tasks {
            guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
                throw MNISTRunnerError.commandBufferFailed("failed to create compute encoder")
            }
            encoder.setComputePipelineState(task.pipeline)
            if task.requiresArgs {
                guard let argsBuffer else {
                    throw MNISTRunnerError.invalidMetadata("kernel \(name) requires args buffer")
                }
                encoder.setBuffer(argsBuffer, offset: 0, index: 0)
                encoder.setBuffer(rootBuffer, offset: 0, index: 1)
            } else {
                encoder.setBuffer(rootBuffer, offset: 0, index: 0)
            }
            let groupWidth = min(task.threadsPerGroup, task.pipeline.maxTotalThreadsPerThreadgroup)
            let threadsPerGroup = MTLSize(width: max(groupWidth, 1), height: 1, depth: 1)
            let threadCount = MTLSize(width: task.threadCount, height: 1, depth: 1)
            encoder.dispatchThreads(threadCount, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }
    }

    private func makeArgsBuffer(value: Float) -> MTLBuffer? {
        var args = Args1(value: value)
        return device.makeBuffer(
            bytes: &args,
            length: MemoryLayout<Args1>.stride,
            options: .storageModeShared
        )
    }

    private func writeTrainSample(index: Int) {
        writeSample(
            imageData: dataset.trainImages,
            labels: dataset.trainLabels,
            index: index
        )
    }

    private func writeTestSample(index: Int) {
        writeSample(
            imageData: dataset.testImages,
            labels: dataset.testLabels,
            index: index
        )
    }

    private func writeSample(imageData: [Float], labels: [UInt8], index: Int) {
        let imageOffset = index * dataset.imageSize
        let imageByteCount = dataset.imageSize * MemoryLayout<Float>.size
        let root = rootBuffer.contents()
        imageData.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            memcpy(root.advanced(by: xOffset), base.advanced(by: imageOffset), imageByteCount)
        }
        root.storeBytes(of: Int32(labels[index]), toByteOffset: labelOffset, as: Int32.self)
    }

    private func readFloat(offset: Int) -> Float {
        rootBuffer.contents().load(fromByteOffset: offset, as: Float.self)
    }

    private func readLogits() -> [Float] {
        var out: [Float] = []
        out.reserveCapacity(numClasses)
        let base = rootBuffer.contents()
        for i in 0..<numClasses {
            out.append(base.load(fromByteOffset: logitsOffset + i * MemoryLayout<Float>.size, as: Float.self))
        }
        return out
    }

    private func argmaxLogits() -> Int {
        argmax(readLogits())
    }

    private func argmax(_ values: [Float]) -> Int {
        var bestIndex = 0
        var bestValue = values[0]
        for i in 1..<values.count {
            if values[i] > bestValue {
                bestValue = values[i]
                bestIndex = i
            }
        }
        return bestIndex
    }

    private func softmaxConfidence(logits: [Float], index: Int) -> Float {
        let maxLogit = logits.max() ?? 0
        var sumExp: Float = 0
        var chosenExp: Float = 0
        for (i, logit) in logits.enumerated() {
            let e = expf(logit - maxLogit)
            sumExp += e
            if i == index {
                chosenExp = e
            }
        }
        if sumExp <= 0 {
            return 0
        }
        return chosenExp / sumExp
    }

    private static func requireFieldOffset(_ name: String, from offsets: [String: Int]) throws -> Int {
        guard let offset = offsets[name] else {
            throw MNISTRunnerError.missingField(name)
        }
        return offset
    }

    private func requireKernel(_ name: String) throws -> [TaskPipeline] {
        guard let tasks = kernelTasks[name], !tasks.isEmpty else {
            throw MNISTRunnerError.missingKernel(name)
        }
        return tasks
    }
}
