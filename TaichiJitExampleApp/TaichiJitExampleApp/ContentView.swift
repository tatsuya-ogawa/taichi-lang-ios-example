//
//  ContentView.swift
//  TaichiJitExampleApp
//
//  Created by Tatsuya Ogawa on 2026/02/11.
//

import SwiftUI
import Combine
import UIKit
import CoreGraphics

private protocol MNISTTrainingRunner: Sendable {
    func train(
        epochs: Int,
        learningRate: Float,
        lossThreshold: Float?,
        progress: ((MNISTTrainingProgress) -> Void)?
    ) throws -> MNISTTrainingSummary

    func inferRandomTestSample() throws -> MNISTInferenceResult
}

extension MNISTMetalRunner: MNISTTrainingRunner {}
extension MNISTMLXRunner: MNISTTrainingRunner {}
extension MNISTMetalRunner: @unchecked Sendable {}
extension MNISTMLXRunner: @unchecked Sendable {}

private struct RunnerInitializationError: LocalizedError, Sendable {
    let message: String

    var errorDescription: String? {
        message
    }
}

enum MNISTTrainingBackend: String, CaseIterable, Identifiable {
    case taichiMetal
    case mlxCustomFunctionManual
    case mlxCustomFunctionAdam

    var id: Self { self }

    var title: String {
        switch self {
        case .taichiMetal:
            return "Taichi Metal"
        case .mlxCustomFunctionManual:
            return "MLX Custom Function (Manual SGD)"
        case .mlxCustomFunctionAdam:
            return "MLX Custom Function (Adam Optimizer)"
        }
    }

    var note: String {
        switch self {
        case .taichiMetal:
            return "Uses Taichi AOT-generated Metal kernels."
        case .mlxCustomFunctionManual:
            return "Uses MLX CustomFunction (Forward + VJP) with handwritten SGD updates."
        case .mlxCustomFunctionAdam:
            return "Uses MLX CustomFunction (Forward + VJP) with MLXOptimizers.Adam updates."
        }
    }
}

@MainActor
final class MNISTViewModel: ObservableObject {

    @Published var output = "Press the Train button to start MNIST training."
    @Published var isRunning = false
    @Published var inferenceImage: UIImage?
    @Published var inferenceLabelText = ""
    @Published var epochs: Int = 10
    @Published var learningRate: Float = 0.001
    @Published var enableLossThreshold = true
    @Published var lossThreshold: Float = 0.10
    @Published var trainingBackend: MNISTTrainingBackend = .taichiMetal

    private let taichiRunnerResult: Result<MNISTMetalRunner, RunnerInitializationError>
    private let mlxManualRunnerResult: Result<MNISTMLXRunner, RunnerInitializationError>
    private let mlxAdamRunnerResult: Result<MNISTMLXRunner, RunnerInitializationError>

    init() {
        do {
            taichiRunnerResult = .success(try MNISTMetalRunner())
        } catch {
            taichiRunnerResult = .failure(
                RunnerInitializationError(message: error.localizedDescription)
            )
        }

        do {
            mlxManualRunnerResult = .success(
                try MNISTMLXRunner(updateRule: .manualSGD)
            )
        } catch {
            mlxManualRunnerResult = .failure(
                RunnerInitializationError(message: error.localizedDescription)
            )
        }

        do {
            mlxAdamRunnerResult = .success(
                try MNISTMLXRunner(updateRule: .adamOptimizer)
            )
        } catch {
            mlxAdamRunnerResult = .failure(
                RunnerInitializationError(message: error.localizedDescription)
            )
        }

        if case .failure(let taichiError) = taichiRunnerResult,
            case .failure(let mlxManualError) = mlxManualRunnerResult,
            case .failure(let mlxAdamError) = mlxAdamRunnerResult
        {
            output = """
            Initialization failed.
            Taichi Metal: \(taichiError.localizedDescription)
            MLX Custom Function (Manual): \(mlxManualError.localizedDescription)
            MLX Custom Function (Adam): \(mlxAdamError.localizedDescription)
            """
        }
    }

    var epochsValidation: String? {
        if epochs < 1 { return "Must be at least 1" }
        if epochs > 100 { return "Recommended max: 100" }
        return nil
    }
    
    var learningRateValidation: String? {
        if learningRate < 0.00001 { return "Must be at least 0.00001" }
        if learningRate > 0.1 { return "Recommended max: 0.1" }
        if learningRate > 0.01 { return "May be too large" }
        return nil
    }
    
    var lossThresholdValidation: String? {
        guard enableLossThreshold else { return nil }
        if lossThreshold < 0 { return "Must be at least 0" }
        if lossThreshold > 1.0 { return "Must be at most 1.0" }
        if lossThreshold < 0.05 { return "May be difficult to achieve" }
        return nil
    }

    func train() {
        guard !isRunning else { return }
        let settings = resolveTrainingSettings()
        let selectedBackend = trainingBackend
        let runnerResult = selectedRunnerResult(for: selectedBackend)
        isRunning = true
        if let threshold = settings.lossThreshold {
            output = String(
                format: "Training started... [%@] (epochs: %d, lr: %.4f, target loss: %.4f)",
                selectedBackend.title,
                settings.epochs,
                settings.learningRate,
                threshold
            )
        } else {
            output = String(
                format: "Training started... [%@] (epochs: %d, lr: %.4f)",
                selectedBackend.title,
                settings.epochs,
                settings.learningRate
            )
        }
        inferenceImage = nil
        inferenceLabelText = ""

        DispatchQueue.global(qos: .userInitiated).async { [runnerResult, selectedBackend, settings] in
            var text = ""
            var summary: MNISTTrainingSummary?
            switch runnerResult {
            case .failure(let error):
                text = "\(selectedBackend.title) initialization failed.\n\(error.localizedDescription)"
            case .success(let runner):
                do {
                    let trained = try runner.train(
                        epochs: settings.epochs,
                        learningRate: settings.learningRate,
                        lossThreshold: settings.lossThreshold
                    ) { progress in
                        let line = String(
                            format: "epoch %d  sample %d/%d  loss %.4f  acc %.3f",
                            progress.epoch,
                            progress.sample,
                            progress.totalSamples,
                            progress.averageLoss,
                            progress.accuracy
                        )
                        DispatchQueue.main.async {
                            self.output = line
                        }
                    }
                    summary = trained
                    let stopLine = trained.stoppedByLossThreshold ? "yes (loss threshold reached)" : "no"
                    text = String(
                        format: """
                        Training finished. [%@]
                        epochs: %d
                        final train loss: %.4f
                        final train acc: %.3f
                        test acc (300 samples): %.3f
                        stopped early: %@
                        """,
                        selectedBackend.title,
                        trained.epochs,
                        trained.trainLoss,
                        trained.trainAccuracy,
                        trained.testAccuracy,
                        stopLine
                    )
                } catch {
                    text = "Training failed.\n\(error.localizedDescription)"
                }
            }

            DispatchQueue.main.async {
                self.output = text
                if summary != nil {
                    self.runInference(backend: selectedBackend)
                }
                self.isRunning = false
            }
        }
    }

    func runInference() {
        runInference(backend: trainingBackend)
    }

    private func runInference(backend: MNISTTrainingBackend) {
        let runnerResult = selectedRunnerResult(for: backend)
        DispatchQueue.global(qos: .userInitiated).async { [runnerResult] in
            switch runnerResult {
            case .failure(let error):
                DispatchQueue.main.async {
                    self.inferenceLabelText = "Inference unavailable: \(error.localizedDescription)"
                }
            case .success(let runner):
                do {
                    let result = try runner.inferRandomTestSample()
                    let image = Self.makeImage(from: result.image, width: 28, height: 28)
                    let label = String(
                        format: "predicted: %d  truth: %d  confidence: %.3f",
                        result.predictedLabel,
                        result.label,
                        result.confidence
                    )
                    DispatchQueue.main.async {
                        self.inferenceImage = image
                        self.inferenceLabelText = label
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.inferenceLabelText = "Inference failed: \(error.localizedDescription)"
                    }
                }
            }
        }
    }

    private func selectedRunnerResult(
        for backend: MNISTTrainingBackend
    ) -> Result<any MNISTTrainingRunner, RunnerInitializationError> {
        switch backend {
        case .taichiMetal:
            return taichiRunnerResult.map { $0 as any MNISTTrainingRunner }
        case .mlxCustomFunctionManual:
            return mlxManualRunnerResult.map { $0 as any MNISTTrainingRunner }
        case .mlxCustomFunctionAdam:
            return mlxAdamRunnerResult.map { $0 as any MNISTTrainingRunner }
        }
    }

    nonisolated private static func makeImage(from pixels: [Float], width: Int, height: Int) -> UIImage? {
        guard pixels.count == width * height else { return nil }
        var rgba = [UInt8](repeating: 0, count: pixels.count * 4)
        for i in 0..<pixels.count {
            let v = max(0, min(255, Int(pixels[i] * 255.0)))
            rgba[4 * i + 0] = UInt8(v)
            rgba[4 * i + 1] = UInt8(v)
            rgba[4 * i + 2] = UInt8(v)
            rgba[4 * i + 3] = 255
        }
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue)
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else {
            return nil
        }
        return UIImage(cgImage: cgImage)
    }

    private struct TrainingSettings {
        let epochs: Int
        let learningRate: Float
        let lossThreshold: Float?
    }

    private func resolveTrainingSettings() -> TrainingSettings {
        let validEpochs = max(1, epochs)
        let validLearningRate = max(0.000001, learningRate)
        let validLossThreshold = enableLossThreshold ? max(0, lossThreshold) : nil

        return TrainingSettings(
            epochs: validEpochs,
            learningRate: validLearningRate,
            lossThreshold: validLossThreshold
        )
    }
}

struct ContentView: View {
    @StateObject private var viewModel = MNISTViewModel()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Training Parameters
                GroupBox {
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Training Parameters")
                            .font(.headline)
                            .foregroundColor(.primary)

                        VStack(alignment: .leading, spacing: 6) {
                            Text("Backend")
                                .font(.subheadline)
                                .fontWeight(.medium)

                            Picker("Backend", selection: $viewModel.trainingBackend) {
                                ForEach(MNISTTrainingBackend.allCases) { backend in
                                    Text(backend.title).tag(backend)
                                }
                            }
                            .pickerStyle(.segmented)

                            Text(viewModel.trainingBackend.note)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Divider()
                        
                        // Epochs
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("Epochs")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                Spacer()
                                Text("\(viewModel.epochs)")
                                    .font(.title3)
                                    .fontWeight(.bold)
                                    .foregroundColor(.blue)
                                    .monospacedDigit()
                            }
                            
                            Stepper("", value: $viewModel.epochs, in: 1...100)
                                .labelsHidden()
                            
                            Text("Number of times to iterate over the dataset")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            if let validation = viewModel.epochsValidation {
                                Label(validation, systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                            }
                        }
                        
                        Divider()
                        
                        // Learning Rate
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text("Learning Rate")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                Spacer()
                                Text(String(format: "%.5f", viewModel.learningRate))
                                    .font(.title3)
                                    .fontWeight(.bold)
                                    .foregroundColor(.blue)
                                    .monospacedDigit()
                            }
                            
                            HStack(spacing: 8) {
                                Button("×0.1") {
                                    viewModel.learningRate *= 0.1
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                                
                                Button("÷2") {
                                    viewModel.learningRate /= 2
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                                
                                Button("×2") {
                                    viewModel.learningRate *= 2
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                                
                                Button("×10") {
                                    viewModel.learningRate *= 10
                                }
                                .buttonStyle(.bordered)
                                .controlSize(.small)
                            }
                            
                            Text("Step size for parameter updates (smaller = more careful)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            if let validation = viewModel.learningRateValidation {
                                Label(validation, systemImage: "exclamationmark.triangle.fill")
                                    .font(.caption)
                                    .foregroundColor(.orange)
                            }
                        }
                    }
                }
                .disabled(viewModel.isRunning)
                
                // Early Stopping
                GroupBox {
                    VStack(alignment: .leading, spacing: 12) {
                        Toggle(isOn: $viewModel.enableLossThreshold) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Early Stopping")
                                    .font(.subheadline)
                                    .fontWeight(.medium)
                                Text("Stop training when target loss is reached")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .toggleStyle(SwitchToggleStyle(tint: .blue))
                        
                        if viewModel.enableLossThreshold {
                            VStack(alignment: .leading, spacing: 6) {
                                HStack {
                                    Text("Target Loss")
                                        .font(.subheadline)
                                    Spacer()
                                    Text(String(format: "%.3f", viewModel.lossThreshold))
                                        .font(.title3)
                                        .fontWeight(.bold)
                                        .foregroundColor(.green)
                                        .monospacedDigit()
                                }
                                
                                Slider(value: $viewModel.lossThreshold, in: 0...1.0, step: 0.01)
                                    .tint(.green)
                                
                                Text("Training will stop when this loss is reached")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                
                                if let validation = viewModel.lossThresholdValidation {
                                    Label(validation, systemImage: "exclamationmark.triangle.fill")
                                        .font(.caption)
                                        .foregroundColor(.orange)
                                }
                            }
                            .padding(.top, 4)
                        }
                    }
                }
                .disabled(viewModel.isRunning)
                
                // Action Buttons
                HStack(spacing: 12) {
                    Button(action: {
                        viewModel.train()
                    }) {
                        Label(
                            viewModel.isRunning ? "Training..." : "Start Training",
                            systemImage: viewModel.isRunning ? "hourglass" : "play.fill"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(viewModel.isRunning)
                    
                    Button(action: {
                        viewModel.runInference()
                    }) {
                        Label("Run Inference", systemImage: "wand.and.stars")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .disabled(viewModel.isRunning)
                }
                
                // Output
                GroupBox {
                    ScrollView {
                        Text(viewModel.output)
                            .font(.system(.callout, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(height: 120)
                } label: {
                    Label("Output", systemImage: "text.alignleft")
                        .font(.subheadline)
                        .fontWeight(.medium)
                }
                
                // Inference Result
                if let image = viewModel.inferenceImage {
                    GroupBox {
                        VStack(spacing: 12) {
                            Image(uiImage: image)
                                .resizable()
                                .interpolation(.none)
                                .frame(width: 140, height: 140)
                                .background(Color.black)
                                .cornerRadius(8)
                                .shadow(radius: 2)
                            
                            if !viewModel.inferenceLabelText.isEmpty {
                                Text(viewModel.inferenceLabelText)
                                    .font(.system(.callout, design: .monospaced))
                                    .multilineTextAlignment(.center)
                            }
                        }
                        .frame(maxWidth: .infinity)
                    } label: {
                        Label("Inference Result", systemImage: "brain")
                            .font(.subheadline)
                            .fontWeight(.medium)
                    }
                }
            }
            .padding()
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
