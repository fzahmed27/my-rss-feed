import SwiftUI

struct SourceDiscoveryView: View {
    @Environment(FeedStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    @State private var websiteURL = ""
    @State private var isSearching = false
    @State private var results: [DiscoveredFeed] = []
    @State private var message: SourceDiscoveryMessage?

    var body: some View {
        NavigationStack {
            List {
                Section("Website") {
                    TextField("https://example.com", text: $websiteURL)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Button {
                        Task { await search() }
                    } label: {
                        Label(isSearching ? "Finding Feeds" : "Find Feeds", systemImage: "magnifyingglass")
                    }
                    .disabled(websiteURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isSearching)
                }

                if isSearching {
                    Section {
                        ProgressView("Checking feed links")
                    }
                }

                if !results.isEmpty {
                    Section("Found Feeds") {
                        ForEach(results) { feed in
                            DiscoveredFeedRow(
                                feed: feed,
                                alreadyAdded: store.sources.contains { $0.url.lowercased() == feed.url.lowercased() },
                                onAdd: {
                                    add(feed)
                                }
                            )
                        }
                    }
                } else if !isSearching, !websiteURL.isEmpty {
                    Section {
                        ContentUnavailableView(
                            "No feeds found yet",
                            systemImage: "antenna.radiowaves.left.and.right"
                        )
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Discover Source")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .alert(item: $message) { message in
                Alert(
                    title: Text(message.title),
                    message: Text(message.body),
                    dismissButton: .default(Text("OK"))
                )
            }
        }
    }

    @MainActor
    private func search() async {
        isSearching = true
        defer { isSearching = false }

        let discovered = await FeedDiscoveryService.discover(from: websiteURL)
        results = discovered

        if discovered.isEmpty {
            message = SourceDiscoveryMessage(
                title: "No Feeds Found",
                body: "No RSS or Atom feed responded for that website."
            )
        }
    }

    private func add(_ feed: DiscoveredFeed) {
        let added = store.addDiscoveredFeed(feed)
        message = SourceDiscoveryMessage(
            title: added ? "Source Added" : "Already Added",
            body: added ? "\(feed.title) is now in Sources." : "\(feed.title) is already in Sources."
        )
    }
}

private struct DiscoveredFeedRow: View {
    let feed: DiscoveredFeed
    let alreadyAdded: Bool
    let onAdd: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "dot.radiowaves.left.and.right")
                .foregroundStyle(Color.accentColor)
                .frame(width: 28)

            VStack(alignment: .leading, spacing: 4) {
                Text(feed.title)
                    .font(.headline)
                Text(feed.url)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            Button(action: onAdd) {
                Image(systemName: alreadyAdded ? "checkmark.circle.fill" : "plus.circle")
                    .frame(width: 36, height: 34)
            }
            .disabled(alreadyAdded)
            .buttonStyle(.borderless)
            .accessibilityLabel(alreadyAdded ? "Feed already added" : "Add feed")
        }
        .padding(.vertical, 4)
    }
}

private struct SourceDiscoveryMessage: Identifiable {
    let id = UUID()
    let title: String
    let body: String
}

#Preview {
    SourceDiscoveryView()
        .environment(FeedStore.preview)
}
