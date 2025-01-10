// Copyright © 2024 Apple Inc.

import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom
import MarkdownUI
import Metal
import SwiftUI
import Tokenizers

struct ContentView: View {

    @State var prompt = ""
    @State var llm = LLMEvaluator()
    @Environment(DeviceStat.self) private var deviceStat

    enum displayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }
    enum modelOption: String, CaseIterable, Identifiable {
        case Qwen32bInstruct4bit,
             Qwen32bInstruct8bit
//             Qwen14bInstruct8bit,
//             Qwen14bInstructBf16,
//             Qwen72bInstruct4bit
             
        var id: Self { self }
    }

    @State private var selectedDisplayStyle = displayStyle.markdown
    
    @State private var selectedModelOption = modelOption.Qwen32bInstruct4bit

    var body: some View {
        VStack(alignment: .leading) {
            VStack {
                HStack {
                    Text(llm.modelInfo)
                        .textFieldStyle(.roundedBorder)

                    Spacer()

                    Text(llm.stat)
                }
                HStack {
                    Spacer()
                    if llm.running {
                        ProgressView()
                            .frame(maxHeight: 20)
                        Spacer()
                    }
                    Picker("", selection: $selectedModelOption) {
                        ForEach(modelOption.allCases, id: \.self) { option in
                            Text(option.rawValue.capitalized)
                                .tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                    #if os(visionOS)
                        .frame(maxWidth: 250)
                    #else
                        .frame(maxWidth: 150)
                    #endif
                    .onChange(of: selectedModelOption) {
                        Task {
                            llm.modelContainer = nil
                            MLX.GPU.clearCache()
                            llm.loadState = .idle
                            llm.setModelConfiguration(modelOption: selectedModelOption.rawValue)
                             _ = try? await llm.load()
                         }
                    }
                    Spacer()
                    Picker("", selection: $selectedDisplayStyle) {
                        ForEach(displayStyle.allCases, id: \.self) { option in
                            Text(option.rawValue.capitalized)
                                .tag(option)
                        }

                    }
                    .pickerStyle(.segmented)
                    #if os(visionOS)
                        .frame(maxWidth: 250)
                    #else
                        .frame(maxWidth: 150)
                    #endif
                }
            }

            // show the model output
            ScrollView(.vertical) {
                ScrollViewReader { sp in
                    Group {
                        if selectedDisplayStyle == .plain {
                            Text(llm.output)
                                .textSelection(.enabled)
                        } else {
                            Markdown(llm.output)
                                .textSelection(.enabled)
                        }
                    }
                    .onChange(of: llm.output) { _, _ in
                        sp.scrollTo("bottom")
                    }

                    Spacer()
                        .frame(width: 1, height: 1)
                        .id("bottom")
                }
            }

            HStack {
                TextField("prompt", text: $prompt)
                    .onSubmit(generate)
                    .disabled(llm.running)
                    #if os(visionOS)
                        .textFieldStyle(.roundedBorder)
                    #endif
                Button("generate", action: generate)
                    .disabled(llm.running)
            }
        }
        #if os(visionOS)
            .padding(40)
        #else
            .padding()
        #endif
        .toolbar {
            ToolbarItem {
                Label(
                    "Memory Usage: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))",
                    systemImage: "info.circle.fill"
                )
                .labelStyle(.titleAndIcon)
                .padding(.horizontal)
                .help(
                    Text(
                        """
                        Active Memory: \(deviceStat.gpuUsage.activeMemory.formatted(.byteCount(style: .memory)))/\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                        Cache Memory: \(deviceStat.gpuUsage.cacheMemory.formatted(.byteCount(style: .memory)))/\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                        Peak Memory: \(deviceStat.gpuUsage.peakMemory.formatted(.byteCount(style: .memory)))
                        """
                    )
                )
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    Task {
                        copyToClipboard(llm.output)
                    }
                } label: {
                    Label("Copy Output", systemImage: "doc.on.doc.fill")
                }
                .disabled(llm.output == "")
                .labelStyle(.titleAndIcon)
            }

        }
        .task {
            self.prompt = llm.modelConfiguration.defaultPrompt

            // pre-load the weights on launch to speed up the first generation
            _ = try? await llm.load()
        }
    }

    private func generate() {
        Task {
//            llm.loadState = .idle
//            llm.setModelConfiguration(modelOption: selectedModelOption.rawValue)
            await llm.generate(prompt: prompt)
        }
    }
    private func copyToClipboard(_ string: String) {
        #if os(macOS)
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(string, forType: .string)
        #else
            UIPasteboard.general.string = string
        #endif
    }
}

@Observable
@MainActor
class LLMEvaluator {

    var running = false

    var output = ""
    var modelInfo = ""
    var stat = ""

    /// This controls which model loads. `phi3_5_4bit` is one of the smaller ones, so this will fit on
    /// more devices.
    var modelConfiguration = ModelRegistry.Qwen32bInstruct4bit
    func setModelConfiguration(modelOption: String) {
        switch  modelOption {
//        case "Qwen14bInstructBf16":
//            modelConfiguration = ModelRegistry.Qwen14bInstructBf16
//        case "Qwen14bInstruct8bit":
//            modelConfiguration = ModelRegistry.Qwen14bInstruct8bit
        case "Qwen32bInstruct8bit":
            modelConfiguration = ModelRegistry.Qwen32bInstruct8bit
        case "Qwen32bInstruct4bit":
            modelConfiguration = ModelRegistry.Qwen32bInstruct4bit
//        case "Qwen72bInstruct4bit":
//            modelConfiguration = ModelRegistry.Qwen72bInstruct4bit
        default:
            modelConfiguration = ModelRegistry.Qwen32bInstruct4bit
        }
    }

    /// parameters controlling the output
    let generateParameters = GenerateParameters(temperature: 0.6)
    let maxTokens = 4000

    /// update the display every N tokens -- 4 looks like it updates continuously
    /// and is low overhead.  observed ~15% reduction in tokens/s when updating
    /// on every token
    let displayEveryNTokens = 4

    enum LoadState {
        case idle
        case loaded(ModelContainer)
    }
    
    var modelContainer: ModelContainer? = nil

    var loadState = LoadState.idle

    /// load and return the model -- can be called multiple times, subsequent calls will
    /// just return the loaded model
    func load() async throws -> ModelContainer {
        switch loadState {
        case .idle:
            // limit the buffer cache, 1 GB
            MLX.GPU.set(cacheLimit: 1024 * 1024 * 1024)

            modelContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: modelConfiguration
            ) {
                [modelConfiguration] progress in
                Task { @MainActor in
                    self.modelInfo =
                        "Downloading \(modelConfiguration.name): \(Int(progress.fractionCompleted * 100))%"
                }
            }
            let numParams = await modelContainer!.perform { context in
                context.model.numParameters()
            }

            self.modelInfo =
            "Loaded \(modelConfiguration.id).  Weights: \(numParams / (1024*1024))M. Path: \(FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!.appending(component: "huggingface").absoluteString)"
            loadState = .loaded(modelContainer!)
            return modelContainer!

        case .loaded(let modelContainer):
            return modelContainer
        }
    }

    func generate(prompt: String) async {
        guard !running else { return }

        running = true
        self.output = ""

        do {
            let modelContainer = try await load()

            // each time you generate you will get something new
            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let result = try await modelContainer.perform { context in
                let input = try await context.processor.prepare(input: .init(prompt: prompt))
                return try MLXLMCommon.generate(
                    input: input, parameters: generateParameters, context: context
                ) { tokens in
                    // update the output -- this will make the view show the text as it generates
                    if tokens.count % displayEveryNTokens == 0 {
                        let text = context.tokenizer.decode(tokens: tokens)
                        Task { @MainActor in
                            self.output = text
                        }
                    }

                    if tokens.count >= maxTokens {
                        return .stop
                    } else {
                        return .more
                    }
                }
            }

            // update the text if needed, e.g. we haven't displayed because of displayEveryNTokens
            if result.output != self.output {
                self.output = result.output
            }
            self.stat = " Tokens/second: \(String(format: "%.3f", result.tokensPerSecond))"

        } catch {
            output = "Failed: \(error)"
        }

        running = false
    }
}
