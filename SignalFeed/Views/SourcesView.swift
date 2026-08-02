import SwiftUI
import UniformTypeIdentifiers

struct SourcesView: View {
    @Environment(FeedStore.self) private var store
    @State private var editorMode: SourceEditorMode?
    @State private var isShowingDiscovery = false
    @State private var isImportingOPML = false
    @State private var exportItem: OPMLExportItem?
    @State private var sourceMessage: SourceActionMessage?

    var body: some View {
        List {
            Section {
                ForEach(store.sources.sorted { $0.name < $1.name }) { source in
                    SourceRow(
                        source: source,
                        health: health(for: source),
                        metrics: store.metrics(for: source),
                        onEdit: {
                            editorMode = SourceEditorMode(source: source)
                        },
                        onToggleEnabled: {
                            store.toggleSourceEnabled(source)
                        },
                        onToggleMuted: {
                            store.toggleSourceMuted(source)
                        }
                    )
                    .swipeActions(edge: .trailing) {
                        Button(role: .destructive) {
                            store.deleteSource(source)
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }

                        Button {
                            editorMode = SourceEditorMode(source: source)
                        } label: {
                            Label("Edit", systemImage: "pencil")
                        }
                        .tint(.blue)

                        Button {
                            store.toggleSourceMuted(source)
                        } label: {
                            Label(source.isMuted ? "Unmute" : "Mute", systemImage: source.isMuted ? "speaker.wave.2" : "speaker.slash")
                        }
                        .tint(.orange)

                        Button {
                            store.toggleSourceEnabled(source)
                        } label: {
                            Label(source.isEnabled ? "Disable" : "Enable", systemImage: source.isEnabled ? "pause" : "play")
                        }
                        .tint(.secondary)
                    }
                }
            } header: {
                Text("Sources")
            } footer: {
                Text("Health comes from the most recent refresh. Set type to Person for AI pioneers, investors, economists, founders, and researchers.")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Sources")
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Menu {
                    Button {
                        isShowingDiscovery = true
                    } label: {
                        Label("Discover Feed", systemImage: "link.badge.plus")
                    }

                    Button {
                        isImportingOPML = true
                    } label: {
                        Label("Import OPML", systemImage: "square.and.arrow.down")
                    }

                    Button {
                        exportOPML()
                    } label: {
                        Label("Export OPML", systemImage: "square.and.arrow.up")
                    }
                    .disabled(store.sources.isEmpty)
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .accessibilityLabel("Source actions")

                Button {
                    editorMode = SourceEditorMode(source: nil)
                } label: {
                    Image(systemName: "plus")
                }
                .accessibilityLabel("Add source")

                Button {
                    Task { await store.refresh() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(store.isLoading)
                .accessibilityLabel("Refresh sources")
            }
        }
        .task {
            await store.refreshIfNeeded()
        }
        .sheet(item: $editorMode) { mode in
            SourceEditorView(source: mode.source)
        }
        .sheet(isPresented: $isShowingDiscovery) {
            SourceDiscoveryView()
        }
        .sheet(item: $exportItem) { item in
            OPMLExportSheet(item: item)
        }
        .fileImporter(
            isPresented: $isImportingOPML,
            allowedContentTypes: [UTType(filenameExtension: "opml") ?? .xml, .xml, .data]
        ) { result in
            importOPML(result)
        }
        .alert(item: $sourceMessage) { message in
            Alert(
                title: Text(message.title),
                message: Text(message.body),
                dismissButton: .default(Text("OK"))
            )
        }
    }

    private func health(for source: FeedSource) -> SourceHealth? {
        store.result?.sourceHealth.first { $0.source.id == source.id }
    }

    private func importOPML(_ result: Result<URL, Error>) {
        switch result {
        case .success(let url):
            let didAccess = url.startAccessingSecurityScopedResource()
            defer {
                if didAccess {
                    url.stopAccessingSecurityScopedResource()
                }
            }

            do {
                let data = try Data(contentsOf: url)
                let count = try store.importOPML(data: data)
                sourceMessage = SourceActionMessage(
                    title: "OPML Imported",
                    body: count == 0 ? "No new sources were added." : "\(count) new sources were added."
                )
            } catch {
                sourceMessage = SourceActionMessage(
                    title: "Import Failed",
                    body: error.localizedDescription
                )
            }
        case .failure(let error):
            sourceMessage = SourceActionMessage(
                title: "Import Failed",
                body: error.localizedDescription
            )
        }
    }

    private func exportOPML() {
        do {
            exportItem = OPMLExportItem(url: try store.makeOPMLExportURL())
        } catch {
            sourceMessage = SourceActionMessage(
                title: "Export Failed",
                body: error.localizedDescription
            )
        }
    }
}

private struct SourceEditorMode: Identifiable {
    let source: FeedSource?

    var id: String {
        source?.id ?? "new"
    }
}

private struct OPMLExportItem: Identifiable {
    let url: URL

    var id: URL { url }
}

private struct SourceActionMessage: Identifiable {
    let id = UUID()
    let title: String
    let body: String
}

private struct OPMLExportSheet: View {
    @Environment(\.dismiss) private var dismiss
    let item: OPMLExportItem

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Image(systemName: "square.and.arrow.up")
                    .font(.system(size: 40, weight: .semibold))
                    .foregroundStyle(Color.accentColor)

                Text(item.url.lastPathComponent)
                    .font(.headline)
                    .multilineTextAlignment(.center)

                ShareLink(item: item.url) {
                    Label("Share OPML", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            }
            .padding(24)
            .navigationTitle("Export Sources")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

private struct SourceRow: View {
    let source: FeedSource
    let health: SourceHealth?
    let metrics: SourceHealthSummary
    let onEdit: () -> Void
    let onToggleEnabled: () -> Void
    let onToggleMuted: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: statusImage)
                    .foregroundStyle(statusColor)
                    .frame(width: 22)

                VStack(alignment: .leading, spacing: 3) {
                    Text(source.name)
                        .font(.headline)
                    Text(source.url)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Spacer()

                Button(action: onEdit) {
                    Image(systemName: "pencil")
                        .frame(width: 32, height: 32)
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Edit \(source.name)")
            }

            HStack(spacing: 8) {
                Label(source.kind.title, systemImage: source.kind.systemImage)
                Text("Rep \(source.reputation.formatted(.number.precision(.fractionLength(2))))")
                if let health {
                    Text("\(health.itemCount) items")
                    Text(health.duration.formatted(.number.precision(.fractionLength(2))) + "s")
                }
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                Label(source.isActive ? "Active" : source.isEnabled ? "Muted" : "Disabled", systemImage: source.isActive ? "play.circle" : "pause.circle")
                Text("Quality \(metrics.averageQuality.formatted(.number.precision(.fractionLength(1))))")
                Text("Failures \(metrics.failureStreak)")
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)

            if let lastSuccessfulFetchAt = metrics.lastSuccessfulFetchAt {
                Label(lastSuccessfulFetchAt.formatted(.relative(presentation: .named)), systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let health, health.status == .failed {
                Text(health.message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            HStack(spacing: 10) {
                Button(action: onToggleEnabled) {
                    Label(source.isEnabled ? "Enabled" : "Disabled", systemImage: source.isEnabled ? "checkmark.circle" : "pause.circle")
                }

                Button(action: onToggleMuted) {
                    Label(source.isMuted ? "Muted" : "Mute", systemImage: source.isMuted ? "speaker.slash.fill" : "speaker.slash")
                }

                Spacer()
            }
            .font(.caption.weight(.semibold))
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 4)
    }

    private var statusImage: String {
        if !source.isActive {
            return SourceStatus.paused.systemImage
        }
        return health?.status.systemImage ?? "questionmark.circle.fill"
    }

    private var statusColor: Color {
        if !source.isActive {
            return .secondary
        }

        switch health?.status {
        case .healthy: return Color.green
        case .empty: return Color.orange
        case .failed: return Color.red
        case .paused: return Color.secondary
        case nil: return Color.secondary
        }
    }
}

#Preview {
    NavigationStack {
        SourcesView()
            .environment(FeedStore.preview)
    }
}
