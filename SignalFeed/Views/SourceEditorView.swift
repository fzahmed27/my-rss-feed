import SwiftUI

struct SourceEditorView: View {
    @Environment(FeedStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    private let source: FeedSource?
    @State private var name: String
    @State private var url: String
    @State private var kind: SourceKind
    @State private var reputation: Double
    @State private var isEnabled: Bool
    @State private var isMuted: Bool

    init(source: FeedSource?) {
        self.source = source
        _name = State(initialValue: source?.name ?? "")
        _url = State(initialValue: source?.url ?? "")
        _kind = State(initialValue: source?.kind ?? .feed)
        _reputation = State(initialValue: source?.reputation ?? 1.0)
        _isEnabled = State(initialValue: source?.isEnabled ?? true)
        _isMuted = State(initialValue: source?.isMuted ?? false)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Source") {
                    TextField("Name", text: $name)
                        .textInputAutocapitalization(.words)

                    TextField("Feed URL", text: $url)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Picker("Type", selection: $kind) {
                        ForEach(SourceKind.allCases) { kind in
                            Label(kind.title, systemImage: kind.systemImage)
                                .tag(kind)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                Section("Controls") {
                    Toggle(isOn: $isEnabled) {
                        Label("Enabled", systemImage: "checkmark.circle")
                    }

                    Toggle(isOn: $isMuted) {
                        Label("Muted", systemImage: "speaker.slash")
                    }
                    .disabled(!isEnabled)
                }

                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Reputation", systemImage: "checkmark.seal")
                            Spacer()
                            Text(reputation.formatted(.number.precision(.fractionLength(2))))
                                .foregroundStyle(.secondary)
                        }

                        Slider(value: $reputation, in: 0.5...1.5, step: 0.05)
                    }
                } header: {
                    Text("Ranking")
                } footer: {
                    Text("Higher reputation gives a small ranking boost. Use it for sources you consistently trust.")
                }
            }
            .navigationTitle(source == nil ? "Add Source" : "Edit Source")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        save()
                    }
                    .disabled(!isValid)
                }
            }
        }
    }

    private var isValid: Bool {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedURL = url.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedName.isEmpty,
              let parsedURL = URL(string: trimmedURL),
              let scheme = parsedURL.scheme?.lowercased(),
              ["http", "https"].contains(scheme),
              parsedURL.host != nil else {
            return false
        }
        return true
    }

    private func save() {
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedURL = url.trimmingCharacters(in: .whitespacesAndNewlines)

        if var source {
            source.name = trimmedName
            source.url = trimmedURL
            source.kind = kind
            source.reputation = reputation
            source.isEnabled = isEnabled
            source.isMuted = isEnabled && isMuted
            store.updateSource(source)
        } else {
            store.addSource(
                name: trimmedName,
                url: trimmedURL,
                kind: kind,
                reputation: reputation,
                isEnabled: isEnabled,
                isMuted: isEnabled && isMuted
            )
        }

        dismiss()
        Task {
            await store.refresh()
        }
    }
}

#Preview {
    SourceEditorView(source: nil)
        .environment(FeedStore.preview)
}
