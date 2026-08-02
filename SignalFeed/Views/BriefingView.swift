import SwiftUI

struct BriefingView: View {
    @Environment(FeedStore.self) private var store
    @State private var safariDestination: SafariDestination?
    @State private var selectedArticle: ScoredArticle?
    @State private var isShowingHistory = false

    var body: some View {
        content
            .navigationTitle("Daily Briefing")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        isShowingHistory = true
                    } label: {
                        Image(systemName: "clock.arrow.circlepath")
                    }
                    .disabled(store.briefingSnapshots.isEmpty)
                    .accessibilityLabel("Briefing history")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await store.refresh() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(store.isLoading)
                    .accessibilityLabel("Refresh briefing")
                }
            }
            .task(id: briefingIdentity) {
                await store.refreshIfNeeded()
                store.saveCurrentBriefingSnapshot()
            }
            .refreshable {
                await store.refresh()
            }
            .sheet(item: $safariDestination) { destination in
                SafariView(url: destination.url)
            }
            .sheet(item: $selectedArticle) { article in
                ArticleDetailView(article: article)
            }
            .sheet(isPresented: $isShowingHistory) {
                BriefingHistoryView()
                    .environment(store)
            }
    }

    private var briefingIdentity: String {
        guard let result = store.result else { return "empty" }
        return [
            result.generatedAt.timeIntervalSinceReferenceDate.description,
            store.presentationSettings.selectedIntent.rawValue,
            String(store.presentationSettings.readingBudgetMinutes)
        ].joined(separator: "|")
    }

    @ViewBuilder
    private var content: some View {
        if let result = store.result {
            briefingList(result: result)
        } else if store.isLoading {
            ProgressView("Building briefing")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "No briefing yet",
                systemImage: "sun.max",
                description: Text("Refresh the feed to build a compact top ten.")
            )
        }
    }

    private func briefingList(result: DigestResult) -> some View {
        let briefing = FounderBriefingGenerator.generate(from: result.articles)

        return List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Five founder signals", systemImage: "sun.max")
                        .font(.headline)
                    Text("Capability, automation, tooling, commercialization, and science signals ranked for an industrial AI founder.")
                        .font(.subheadline)
                        .foregroundStyle(.primary)
                    Label("\(store.presentationSettings.readingBudgetMinutes) minute reading budget", systemImage: "timer")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Generated \(result.generatedAt.formatted(date: .abbreviated, time: .shortened)) from \(result.fetchedCount) fetched items.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    if let snapshot = store.currentBriefingSnapshot {
                        if let completion = store.currentBriefingCompletion {
                            Label(
                                "Completed \(completion.completedAt.formatted(date: .abbreviated, time: .shortened))",
                                systemImage: "checkmark.circle.fill"
                            )
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.green)
                        } else {
                            Button {
                                store.completeCurrentBriefing()
                            } label: {
                                Label("Complete review", systemImage: "checkmark.circle")
                                    .font(.subheadline.weight(.semibold))
                                    .frame(minHeight: 44)
                            }
                            .buttonStyle(.borderedProminent)
                            .accessibilityHint("Stores the completion date and current intent")
                        }

                        HStack(spacing: 16) {
                            Label("Saved on device", systemImage: "externaldrive.fill.badge.checkmark")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            ShareLink(item: snapshot.exportText) {
                                Label("Share briefing", systemImage: "square.and.arrow.up")
                                    .font(.subheadline.weight(.semibold))
                                    .frame(minHeight: 44)
                            }
                            .simultaneousGesture(TapGesture().onEnded {
                                store.recordBriefingShare(id: snapshot.id)
                            })
                        }
                    } else {
                        Label("Saving briefing", systemImage: "arrow.trianglehead.2.clockwise.rotate.90")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 4)
            }

            if result.articles.isEmpty {
                ContentUnavailableView(
                    "No high-signal items",
                    systemImage: "line.3.horizontal.decrease.circle",
                    description: Text("Try lowering the minimum score or widening the date window.")
                )
            } else {
                ForEach(briefing) { item in
                    Section {
                        if let article = item.article {
                            BriefingRow(
                                rank: item.rank,
                                slotTitle: item.slot.title,
                                article: article,
                                isBookmarked: store.isBookmarked(article),
                                interaction: store.interaction(for: article),
                                onSelect: {
                                    selectedArticle = article
                                },
                                onOpen: {
                                    guard let url = URL(string: article.article.link) else { return }
                                    store.recordOpen(for: article)
                                    safariDestination = SafariDestination(url: url)
                                },
                                onToggleBookmark: {
                                    store.toggleBookmark(for: article)
                                },
                                onFeedback: { feedback, reasons in
                                    store.setFeedback(feedback, reasons: reasons, for: article)
                                }
                            )
                        } else {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("No strong item")
                                    .font(.subheadline.weight(.semibold))
                                Text(item.emptyReason)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.vertical, 6)
                        }
                    } header: {
                        Label(item.slot.title, systemImage: item.slot.systemImage)
                    } footer: {
                        Text(item.slot.subtitle)
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
    }
}

private struct BriefingHistoryView: View {
    @Environment(FeedStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if store.briefingSnapshots.isEmpty {
                    ContentUnavailableView(
                        "No saved briefings",
                        systemImage: "clock.arrow.circlepath",
                        description: Text("Generated briefings will be saved here automatically.")
                    )
                } else {
                    List(store.briefingSnapshots) { snapshot in
                        NavigationLink {
                            BriefingSnapshotView(snapshotID: snapshot.id)
                        } label: {
                            VStack(alignment: .leading, spacing: 6) {
                                Text(snapshot.savedAt.formatted(date: .abbreviated, time: .shortened))
                                    .font(.headline)
                                Text(snapshot.intent.title)
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                                HStack(spacing: 12) {
                                    Label("\(snapshot.readingBudgetMinutes) min", systemImage: "timer")
                                    Label("\(snapshot.items.count) slots", systemImage: "list.number")
                                }
                                .font(.caption)
                                .foregroundStyle(.secondary)

                                if let completion = store.completion(for: snapshot.id) {
                                    Label(
                                        "Completed \(completion.completedAt.formatted(date: .abbreviated, time: .shortened))",
                                        systemImage: "checkmark.circle.fill"
                                    )
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.green)
                                }
                            }
                            .padding(.vertical, 4)
                        }
                        .accessibilityLabel("Briefing saved \(snapshot.savedAt.formatted()), \(snapshot.intent.title), \(snapshot.items.count) slots")
                    }
                }
            }
            .navigationTitle("Briefing History")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}

private struct BriefingSnapshotView: View {
    @Environment(FeedStore.self) private var store
    let snapshotID: UUID

    private var snapshot: FounderBriefingSnapshot? {
        store.briefingSnapshots.first { $0.id == snapshotID }
    }

    var body: some View {
        Group {
            if let snapshot {
                List {
                    Section {
                        LabeledContent("Generated", value: snapshot.generatedAt.formatted(date: .abbreviated, time: .shortened))
                        LabeledContent("Intent", value: snapshot.intent.title)
                        LabeledContent("Reading budget", value: "\(snapshot.readingBudgetMinutes) minutes")
                        LabeledContent("Source items", value: "\(snapshot.fetchedCount)")
                        if let completion = store.completion(for: snapshot.id) {
                            LabeledContent(
                                "Completed",
                                value: completion.completedAt.formatted(date: .abbreviated, time: .shortened)
                            )
                            LabeledContent("Reviewed", value: "\(completion.reviewedSlotCount) of \(snapshot.items.count) slots")
                        }
                        if snapshot.shareCount > 0 {
                            LabeledContent("Shared", value: "\(snapshot.shareCount) time\(snapshot.shareCount == 1 ? "" : "s")")
                        }

                        ShareLink(item: snapshot.exportText) {
                            Label("Share saved briefing", systemImage: "square.and.arrow.up")
                                .frame(minHeight: 44)
                        }
                        .simultaneousGesture(TapGesture().onEnded {
                            store.recordBriefingShare(id: snapshot.id)
                        })
                    }

                    ForEach(snapshot.items) { item in
                        Section {
                            if let article = item.article {
                                VStack(alignment: .leading, spacing: 10) {
                                    Text(article.article.title)
                                        .font(.headline)
                                    briefingDetail("What happened", article.decisionSummary.whatChanged)
                                    briefingDetail("Why it matters", article.decisionSummary.whyItMattersToYou)
                                    briefingDetail("What you should do", article.decisionSummary.suggestedAction)
                                }
                                .padding(.vertical, 4)
                            } else {
                                Text(item.emptyReason)
                                    .foregroundStyle(.secondary)
                            }
                        } header: {
                            Label(item.slot.title, systemImage: item.slot.systemImage)
                        }
                    }
                }
            } else {
                ContentUnavailableView("Briefing unavailable", systemImage: "exclamationmark.triangle")
            }
        }
        .navigationTitle("Saved Briefing")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func briefingDetail(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline)
        }
    }
}

private struct BriefingRow: View {
    let rank: Int
    let slotTitle: String
    let article: ScoredArticle
    let isBookmarked: Bool
    let interaction: ArticleInteraction
    let onSelect: () -> Void
    let onOpen: () -> Void
    let onToggleBookmark: () -> Void
    let onFeedback: (ArticleFeedback, [FeedbackReason]) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top, spacing: 12) {
                    Text("\(rank)")
                        .font(.headline.weight(.bold))
                        .foregroundStyle(.white)
                        .frame(width: 34, height: 34)
                        .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 8))

                    VStack(alignment: .leading, spacing: 5) {
                        Text(slotTitle)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Text(article.article.title)
                            .font(.subheadline.weight(.semibold))
                            .lineLimit(3)

                        HStack(spacing: 8) {
                            Text(article.article.sourceName)
                            Text(article.relevanceLabel)
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    }

                    Spacer(minLength: 0)
                }

                DecisionLine(title: "What happened", value: article.decisionSummary.whatChanged)
                DecisionLine(title: "Why it matters", value: article.decisionSummary.whyItMattersToYou)
                DecisionLine(title: "Should I care?", value: article.decisionSummary.shouldCare)
                DecisionLine(title: "What you should do", value: article.decisionSummary.suggestedAction)
                HStack(spacing: 8) {
                    Label(article.decisionSummary.confidence.title, systemImage: "checkmark.shield")
                    Label("\(article.decisionSummary.estimatedReadingMinutes) min", systemImage: "timer")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .contentShape(Rectangle())
            .onTapGesture(perform: onSelect)

            HStack(spacing: 10) {
                Button(action: onToggleBookmark) {
                    Image(systemName: isBookmarked ? "bookmark.fill" : "bookmark")
                        .frame(width: 30, height: 28)
                }
                .accessibilityLabel(isBookmarked ? "Remove bookmark" : "Bookmark article")

                Button {
                    onFeedback(.liked, [])
                } label: {
                    Image(systemName: interaction.feedback == .liked ? "hand.thumbsup.fill" : "hand.thumbsup")
                        .frame(width: 30, height: 28)
                }
                .accessibilityLabel("Mark useful")

                Button {
                    onFeedback(.disliked, [])
                } label: {
                    Image(systemName: interaction.feedback == .disliked ? "hand.thumbsdown.fill" : "hand.thumbsdown")
                        .frame(width: 30, height: 28)
                }
                .accessibilityLabel("Mark not useful")

                Spacer()

                Button(action: onOpen) {
                    Image(systemName: "safari")
                        .frame(width: 30, height: 28)
                }
                .accessibilityLabel("Open article")
            }
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 4)
    }
}

private struct DecisionLine: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
    }
}

#Preview {
    NavigationStack {
        BriefingView()
            .environment(FeedStore.preview)
    }
}
