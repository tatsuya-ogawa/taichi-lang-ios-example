//
//  MNISTMLXRunner.swift
//  TaichiJitExampleApp
//
//  Created by Codex on 2026/02/23.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

enum MNISTMLXRunnerError: LocalizedError {
    case missingBundleResource(String)
    case invalidDatasetFormat(String)
    case invalidKernelSpec(String)
    case invalidPrediction(String)

    var errorDescription: String? {
        switch self {
        case .missingBundleResource(let name):
            return "Missing app resource: \(name)"
        case .invalidDatasetFormat(let detail):
            return "Invalid MNIST dataset format: \(detail)"
        case .invalidKernelSpec(let detail):
            return "Invalid Slang kernel spec: \(detail)"
        case .invalidPrediction(let detail):
            return "Invalid model prediction: \(detail)"
        }
    }
}

enum MNISTMLXUpdateRule: Sendable {
    case manualSGD
    case adamOptimizer
}

final class MNISTMLXRunner {
    nonisolated private final class LinearMNISTModel: Module, @unchecked Sendable {
        var weights: MLXArray
        var bias: MLXArray
        private let linearWithBias: ([MLXArray]) -> [MLXArray]

        init(
            weights: MLXArray,
            bias: MLXArray,
            linearWithBias: @escaping ([MLXArray]) -> [MLXArray]
        ) {
            self.weights = weights
            self.bias = bias
            self.linearWithBias = linearWithBias
            super.init()
        }

        func logits(_ x: MLXArray) -> MLXArray {
            linearWithBias([x, weights, bias])[0]
        }
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
                throw MNISTMLXRunnerError.invalidDatasetFormat("file too small")
            }

            var offset = 0
            let magic = String(data: data[offset..<offset + 4], encoding: .ascii) ?? ""
            offset += 4
            guard magic == "MNST" else {
                throw MNISTMLXRunnerError.invalidDatasetFormat("bad magic: \(magic)")
            }

            let version = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
            offset += 4
            guard version == 1 else {
                throw MNISTMLXRunnerError.invalidDatasetFormat("unsupported version: \(version)")
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
                throw MNISTMLXRunnerError.invalidDatasetFormat("byte size mismatch")
            }

            let trainImages: [Float] = data[offset..<offset + trainImageBytes].withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            offset += trainImageBytes

            let trainLabels: [UInt8] = Array(data[offset..<offset + trainLabelBytes])
            offset += trainLabelBytes

            let testImages: [Float] = data[offset..<offset + testImageBytes].withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            offset += testImageBytes

            let testLabels: [UInt8] = Array(data[offset..<offset + testLabelBytes])

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

    private struct SlangKernelSpec: Decodable {
        let kernelName: String
        let inputNames: [String]
        let outputNames: [String]
        let source: String
        let header: String

        private enum CodingKeys: String, CodingKey {
            case kernelName = "kernel_name"
            case inputNames = "input_names"
            case outputNames = "output_names"
            case source
            case header
        }
    }

    private struct KernelSpecLibrary: Decodable {
        let kernels: [String: SlangKernelSpec]
    }

    private let dataset: MNISTBinaryDataset
    private let trainLabels: [Int32]
    private let testLabels: [Int32]
    private let linearWithBias: ([MLXArray]) -> [MLXArray]
    private let featureTransform: (MLXArray) -> MLXArray
    private let updateRule: MNISTMLXUpdateRule

    private var weights: MLXArray
    private var bias: MLXArray

    private let numClasses = 10
    private let batchSize = 128

    init(updateRule: MNISTMLXUpdateRule = .manualSGD) throws {
        self.updateRule = updateRule
        let bundle = Bundle.main
        guard let datasetURL = bundle.url(forResource: "mnist_subset", withExtension: "bin") else {
            throw MNISTMLXRunnerError.missingBundleResource("MNIST/mnist_subset.bin")
        }
        let slangFeatureTransformSpec = try Self.loadKernelSpec(
            named: "run_backward_custom_mlx",
            from: bundle
        )
        try Self.validateKernelSpec(
            slangFeatureTransformSpec,
            expectedInputCount: 1,
            expectedOutputCount: 1,
            label: "feature transform"
        )
        let linearKernelLibrary = try Self.loadLinearKernelLibrary(from: bundle)
        let forwardSpec = try Self.requiredKernel(
            named: "mnist_linear_forward",
            in: linearKernelLibrary
        )
        try Self.validateKernelSpec(
            forwardSpec,
            expectedInputCount: 3,
            expectedOutputCount: 1,
            label: "mnist_linear_forward"
        )
        let gradXSpec = try Self.requiredKernel(
            named: "mnist_linear_grad_x",
            in: linearKernelLibrary
        )
        try Self.validateKernelSpec(
            gradXSpec,
            expectedInputCount: 2,
            expectedOutputCount: 1,
            label: "mnist_linear_grad_x"
        )
        let gradWSpec = try Self.requiredKernel(
            named: "mnist_linear_grad_w",
            in: linearKernelLibrary
        )
        try Self.validateKernelSpec(
            gradWSpec,
            expectedInputCount: 2,
            expectedOutputCount: 1,
            label: "mnist_linear_grad_w"
        )
        let gradBSpec = try Self.requiredKernel(
            named: "mnist_linear_grad_b",
            in: linearKernelLibrary
        )
        try Self.validateKernelSpec(
            gradBSpec,
            expectedInputCount: 1,
            expectedOutputCount: 1,
            label: "mnist_linear_grad_b"
        )

        let dataset = try MNISTBinaryDataset.load(from: datasetURL)
        guard dataset.imageSize == 28 * 28 else {
            throw MNISTMLXRunnerError.invalidDatasetFormat("unexpected image size: \(dataset.imageSize)")
        }
        self.dataset = dataset

        self.trainLabels = dataset.trainLabels.map(Int32.init)
        self.testLabels = dataset.testLabels.map(Int32.init)

        let weightScale: Float = 0.01
        let initialWeights = (0..<(numClasses * dataset.imageSize)).map { _ in
            Float.random(in: -weightScale...weightScale)
        }
        self.weights = MLXArray(initialWeights, [numClasses, dataset.imageSize])
        self.bias = MLXArray.zeros([numClasses], type: Float.self)

        // Slang-generated kernel spec loaded from app bundle.
        // This computes d(x^2)/dx = 2x elementwise and is wired before the linear layer.
        let slangFeatureTransformKernel = MLXFast.metalKernel(
            name: slangFeatureTransformSpec.kernelName,
            inputNames: slangFeatureTransformSpec.inputNames,
            outputNames: slangFeatureTransformSpec.outputNames,
            source: slangFeatureTransformSpec.source,
            header: slangFeatureTransformSpec.header
        )
        self.featureTransform = { [slangFeatureTransformKernel] x in
            let batch = x.shape[0]
            let inFeatures = x.shape[1]
            let total = batch * inFeatures
            return slangFeatureTransformKernel(
                [x],
                grid: (total, 1, 1),
                threadGroup: (128, 1, 1),
                outputShapes: [[batch, inFeatures]],
                outputDTypes: [x.dtype]
            )[0]
        }

        let forwardKernel = MLXFast.metalKernel(
            name: forwardSpec.kernelName,
            inputNames: forwardSpec.inputNames,
            outputNames: forwardSpec.outputNames,
            source: forwardSpec.source,
            header: forwardSpec.header
        )

        let gradXKernel = MLXFast.metalKernel(
            name: gradXSpec.kernelName,
            inputNames: gradXSpec.inputNames,
            outputNames: gradXSpec.outputNames,
            source: gradXSpec.source,
            header: gradXSpec.header
        )

        let gradWKernel = MLXFast.metalKernel(
            name: gradWSpec.kernelName,
            inputNames: gradWSpec.inputNames,
            outputNames: gradWSpec.outputNames,
            source: gradWSpec.source,
            header: gradWSpec.header
        )

        let gradBKernel = MLXFast.metalKernel(
            name: gradBSpec.kernelName,
            inputNames: gradBSpec.inputNames,
            outputNames: gradBSpec.outputNames,
            source: gradBSpec.source,
            header: gradBSpec.header
        )

        self.linearWithBias = CustomFunction {
            Forward { [forwardKernel] inputs in
                let x = inputs[0]
                let w = inputs[1]
                let b = inputs[2]
                let batch = x.shape[0]
                let outFeatures = w.shape[0]
                let outputShape = [batch, outFeatures]
                let total = batch * outFeatures

                return forwardKernel(
                    [x, w, b],
                    grid: (total, 1, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [outputShape],
                    outputDTypes: [x.dtype]
                )
            }

            VJP { [gradXKernel, gradWKernel, gradBKernel] primals, cotangents in
                let x = primals[0]
                let w = primals[1]
                let cotangent = cotangents[0]
                let batch = x.shape[0]
                let inFeatures = x.shape[1]
                let outFeatures = w.shape[0]

                let gradX = gradXKernel(
                    [w, cotangent],
                    grid: (batch * inFeatures, 1, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [[batch, inFeatures]],
                    outputDTypes: [x.dtype]
                )[0]

                let gradW = gradWKernel(
                    [x, cotangent],
                    grid: (outFeatures * inFeatures, 1, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [[outFeatures, inFeatures]],
                    outputDTypes: [w.dtype]
                )[0]

                let gradB = gradBKernel(
                    [cotangent],
                    grid: (outFeatures, 1, 1),
                    threadGroup: (128, 1, 1),
                    outputShapes: [[outFeatures]],
                    outputDTypes: [w.dtype]
                )[0]

                return [gradX, gradW, gradB]
            }
        }

        eval(weights, bias)
    }

    func train(
        epochs: Int,
        learningRate: Float,
        lossThreshold: Float?,
        progress: ((MNISTTrainingProgress) -> Void)?
    ) throws -> MNISTTrainingSummary {
        switch updateRule {
        case .manualSGD:
            return try trainWithManualSGD(
                epochs: epochs,
                learningRate: learningRate,
                lossThreshold: lossThreshold,
                progress: progress
            )
        case .adamOptimizer:
            return try trainWithAdam(
                epochs: epochs,
                learningRate: learningRate,
                lossThreshold: lossThreshold,
                progress: progress
            )
        }
    }

    private func trainWithManualSGD(
        epochs: Int,
        learningRate: Float,
        lossThreshold: Float?,
        progress: ((MNISTTrainingProgress) -> Void)?
    ) throws -> MNISTTrainingSummary {
        let targetEpochs = max(1, epochs)
        let trainCount = dataset.trainCount

        let featureTransform = self.featureTransform
        let lossAndGrad = valueAndGrad(
            { [linearWithBias] args in
                let x = args[0]
                let y = args[1]
                let w = args[2]
                let b = args[3]
                let logits = linearWithBias([x, w, b])[0]
                let loss = Self.crossEntropyLoss(logits: logits, targets: y)
                return [loss]
            },
            argumentNumbers: [2, 3]
        )

        var completedEpochs = 0
        var stoppedByLossThreshold = false
        var lastAvgLoss: Float = 0
        var lastAvgAccuracy: Float = 0

        for epoch in 0..<targetEpochs {
            var seen = 0
            var weightedLossSum: Float = 0
            var weightedAccSum: Float = 0
            var nextProgressSample = 100

            while seen < trainCount {
                let count = min(batchSize, trainCount - seen)
                let xBatch = makeFeatureArray(from: dataset.trainImages, start: seen, count: count)
                let xBatchTransformed = featureTransform(xBatch)
                let yBatch = makeLabelArray(from: trainLabels, start: seen, count: count)

                let (lossOutputs, grads) = lossAndGrad([xBatchTransformed, yBatch, weights, bias])
                let batchLoss = lossOutputs[0]

                weights = weights - learningRate * grads[0]
                bias = bias - learningRate * grads[1]

                let logits = linearWithBias([xBatchTransformed, weights, bias])[0]
                let predicted = argMax(logits, axis: -1)
                let batchAccuracy = mean((predicted .== yBatch).asType(.float32))

                eval(weights, bias, batchLoss, batchAccuracy)

                let batchLossValue = batchLoss.item(Float.self)
                let batchAccuracyValue = batchAccuracy.item(Float.self)
                weightedLossSum += batchLossValue * Float(count)
                weightedAccSum += batchAccuracyValue * Float(count)
                seen += count

                if seen >= nextProgressSample || seen == trainCount {
                    let avgLoss = weightedLossSum / Float(seen)
                    let avgAccuracy = weightedAccSum / Float(seen)
                    progress?(
                        MNISTTrainingProgress(
                            epoch: epoch + 1,
                            sample: seen,
                            totalSamples: trainCount,
                            averageLoss: avgLoss,
                            accuracy: avgAccuracy
                        )
                    )
                    lastAvgLoss = avgLoss
                    lastAvgAccuracy = avgAccuracy
                    nextProgressSample += 100

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
            trainAccuracy: lastAvgAccuracy,
            testAccuracy: testAccuracy,
            stoppedByLossThreshold: stoppedByLossThreshold
        )
    }

    private func trainWithAdam(
        epochs: Int,
        learningRate: Float,
        lossThreshold: Float?,
        progress: ((MNISTTrainingProgress) -> Void)?
    ) throws -> MNISTTrainingSummary {
        let targetEpochs = max(1, epochs)
        let trainCount = dataset.trainCount

        let model = LinearMNISTModel(
            weights: weights,
            bias: bias,
            linearWithBias: linearWithBias
        )
        let optimizer = Adam(learningRate: learningRate)
        let featureTransform = self.featureTransform

        let lossAndGrad = valueAndGrad(model: model) { model, args in
            let x = args[0]
            let y = args[1]
            let logits = model.logits(x)
            let loss = Self.crossEntropyLoss(logits: logits, targets: y)
            return [loss]
        }

        var completedEpochs = 0
        var stoppedByLossThreshold = false
        var lastAvgLoss: Float = 0
        var lastAvgAccuracy: Float = 0

        for epoch in 0..<targetEpochs {
            var seen = 0
            var weightedLossSum: Float = 0
            var weightedAccSum: Float = 0
            var nextProgressSample = 100

            while seen < trainCount {
                let count = min(batchSize, trainCount - seen)
                let xBatch = makeFeatureArray(from: dataset.trainImages, start: seen, count: count)
                let xBatchTransformed = featureTransform(xBatch)
                let yBatch = makeLabelArray(from: trainLabels, start: seen, count: count)

                let (lossOutputs, grads) = lossAndGrad(model, [xBatchTransformed, yBatch])
                let batchLoss = lossOutputs[0]

                optimizer.update(model: model, gradients: grads)

                let logits = model.logits(xBatchTransformed)
                let predicted = argMax(logits, axis: -1)
                let batchAccuracy = mean((predicted .== yBatch).asType(.float32))

                eval(model, optimizer, batchLoss, batchAccuracy)

                let batchLossValue = batchLoss.item(Float.self)
                let batchAccuracyValue = batchAccuracy.item(Float.self)
                weightedLossSum += batchLossValue * Float(count)
                weightedAccSum += batchAccuracyValue * Float(count)
                seen += count

                if seen >= nextProgressSample || seen == trainCount {
                    let avgLoss = weightedLossSum / Float(seen)
                    let avgAccuracy = weightedAccSum / Float(seen)
                    progress?(
                        MNISTTrainingProgress(
                            epoch: epoch + 1,
                            sample: seen,
                            totalSamples: trainCount,
                            averageLoss: avgLoss,
                            accuracy: avgAccuracy
                        )
                    )
                    lastAvgLoss = avgLoss
                    lastAvgAccuracy = avgAccuracy
                    nextProgressSample += 100

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

        weights = model.weights
        bias = model.bias
        eval(weights, bias)

        let testAccuracy = try evaluateTestAccuracy(sampleLimit: min(dataset.testCount, 300))
        return MNISTTrainingSummary(
            epochs: completedEpochs,
            trainLoss: lastAvgLoss,
            trainAccuracy: lastAvgAccuracy,
            testAccuracy: testAccuracy,
            stoppedByLossThreshold: stoppedByLossThreshold
        )
    }

    func inferRandomTestSample() throws -> MNISTInferenceResult {
        let index = Int.random(in: 0..<dataset.testCount)
        return try inferTestSample(index: index)
    }

    private func inferTestSample(index: Int) throws -> MNISTInferenceResult {
        let imageStart = index * dataset.imageSize
        let imageEnd = imageStart + dataset.imageSize
        let image = Array(dataset.testImages[imageStart..<imageEnd])

        let x = MLXArray(image, [1, dataset.imageSize])
        let transformedX = featureTransform(x)
        let logits = linearWithBias([transformedX, weights, bias])[0]
        let probabilities = softmax(logits, axis: -1)
        let predictedArray = argMax(logits, axis: -1)
        eval(probabilities, predictedArray)

        let predictedLabel = Int(predictedArray.item(Int32.self))
        let confidenceCandidates = probabilities.reshaped([numClasses]).asArray(Float.self)
        guard confidenceCandidates.indices.contains(predictedLabel) else {
            throw MNISTMLXRunnerError.invalidPrediction("predicted label out of range: \(predictedLabel)")
        }

        return MNISTInferenceResult(
            image: image,
            label: Int(dataset.testLabels[index]),
            predictedLabel: predictedLabel,
            confidence: confidenceCandidates[predictedLabel]
        )
    }

    private func evaluateTestAccuracy(sampleLimit: Int) throws -> Float {
        let total = max(1, min(sampleLimit, dataset.testCount))
        var seen = 0
        var correct: Float = 0

        while seen < total {
            let count = min(batchSize, total - seen)
            let xBatch = makeFeatureArray(from: dataset.testImages, start: seen, count: count)
            let xBatchTransformed = featureTransform(xBatch)
            let yBatch = makeLabelArray(from: testLabels, start: seen, count: count)

            let logits = linearWithBias([xBatchTransformed, weights, bias])[0]
            let predicted = argMax(logits, axis: -1)
            let batchCorrect = sum((predicted .== yBatch).asType(.float32))
            eval(batchCorrect)

            correct += batchCorrect.item(Float.self)
            seen += count
        }

        return correct / Float(total)
    }

    private static func kernelJSONURL(named name: String, in bundle: Bundle) -> URL? {
        bundle.url(forResource: name, withExtension: "json", subdirectory: "Slang")
            ?? bundle.url(forResource: name, withExtension: "json")
    }

    private static func loadKernelSpec(named name: String, from bundle: Bundle) throws -> SlangKernelSpec {
        guard let url = kernelJSONURL(named: name, in: bundle) else {
            throw MNISTMLXRunnerError.missingBundleResource(
                "Slang/\(name).json (or bundle root \(name).json)"
            )
        }
        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(SlangKernelSpec.self, from: data)
        } catch {
            throw MNISTMLXRunnerError.invalidKernelSpec(
                "failed to decode \(name).json: \(error.localizedDescription)"
            )
        }
    }

    private static func loadLinearKernelLibrary(from bundle: Bundle) throws -> [String: SlangKernelSpec] {
        let name = "mnist_linear_kernels"
        guard let url = kernelJSONURL(named: name, in: bundle) else {
            throw MNISTMLXRunnerError.missingBundleResource(
                "Slang/\(name).json (or bundle root \(name).json)"
            )
        }
        do {
            let data = try Data(contentsOf: url)
            let library = try JSONDecoder().decode(KernelSpecLibrary.self, from: data)
            return library.kernels
        } catch {
            throw MNISTMLXRunnerError.invalidKernelSpec(
                "failed to decode \(name).json: \(error.localizedDescription)"
            )
        }
    }

    private static func requiredKernel(
        named name: String,
        in library: [String: SlangKernelSpec]
    ) throws -> SlangKernelSpec {
        guard let spec = library[name] else {
            throw MNISTMLXRunnerError.invalidKernelSpec("kernel '\(name)' not found in mnist_linear_kernels.json")
        }
        return spec
    }

    private static func validateKernelSpec(
        _ spec: SlangKernelSpec,
        expectedInputCount: Int,
        expectedOutputCount: Int,
        label: String
    ) throws {
        guard spec.inputNames.count == expectedInputCount else {
            throw MNISTMLXRunnerError.invalidKernelSpec(
                "\(label): expected \(expectedInputCount) input(s), got \(spec.inputNames.count)"
            )
        }
        guard spec.outputNames.count == expectedOutputCount else {
            throw MNISTMLXRunnerError.invalidKernelSpec(
                "\(label): expected \(expectedOutputCount) output(s), got \(spec.outputNames.count)"
            )
        }
    }

    private static func crossEntropyLoss(logits: MLXArray, targets: MLXArray) -> MLXArray {
        let selectedLogit = takeAlong(
            logits,
            targets.expandedDimensions(axis: -1),
            axis: -1
        ).squeezed(axis: -1)
        return mean(logSumExp(logits, axis: -1) - selectedLogit)
    }

    private func makeFeatureArray(from imageData: [Float], start: Int, count: Int) -> MLXArray {
        let lower = start * dataset.imageSize
        let upper = lower + count * dataset.imageSize
        return MLXArray(Array(imageData[lower..<upper]), [count, dataset.imageSize])
    }

    private func makeLabelArray(from labels: [Int32], start: Int, count: Int) -> MLXArray {
        MLXArray(Array(labels[start..<start + count]), [count])
    }
}
