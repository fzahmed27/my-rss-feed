import SwiftUI

struct OpportunityInboxView: View {
    @Environment(FeedStore.self) private var store
    @State private var filter: OpportunityInboxFilter = .all
    @State private var showsResearchQueue = false
    @State private var safariDestination: SafariDestination?
    @State private var selectedArticle: ScoredArticle?

    var body: some View {
        List {
            Section {
                OpportunityInboxHeader(
                    totalCount: store.opportunityHypotheses.count,
                    savedCount: store.savedResearchHypotheses.count,
                    counts: labelCounts
                )
                .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                .listRowSeparator(.hidden)

                Picker("Filter", selection: $filter) {
                    ForEach(OpportunityInboxFilter.allCases) { filter in
                        Text(filter.title).tag(filter)
                    }
                }
                .pickerStyle(.segmented)
                .listRowInsets(EdgeInsets(top: 0, leading: 16, bottom: 8, trailing: 16))
                .listRowSeparator(.hidden)
            }

            if filteredHypotheses.isEmpty {
                ContentUnavailableView(
                    showsResearchQueue ? "Research queue is empty" : "No opportunity hypotheses",
                    systemImage: showsResearchQueue ? "tray" : "sparkles",
                    description: Text(
                        showsResearchQueue
                            ? "Save a hypothesis to collect it here."
                            : "Refresh the feed or widen ranking settings to surface testable candidates."
                    )
                )
                .listRowSeparator(.hidden)
            } else {
                ForEach(filteredHypotheses) { hypothesis in
                    OpportunityHypothesisCard(
                        hypothesis: hypothesis,
                        onToggleQueue: {
                            store.toggleResearchQueue(id: hypothesis.id)
                        },
                        onViewEvidence: {
                            selectedArticle = hypothesis.supportingArticle
                        },
                        onOpenSource: {
                            open(hypothesis.supportingArticle)
                        }
                    )
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                    .listRowSeparator(.hidden)
                }
            }
        }
        .listStyle(.plain)
        .environment(\.defaultMinListRowHeight, 1)
        .navigationTitle(showsResearchQueue ? "Research Queue" : "Opportunity Inbox")
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button {
                    showsResearchQueue.toggle()
                } label: {
                    Image(systemName: showsResearchQueue ? "tray.full.fill" : "tray.full")
                }
                .accessibilityLabel(showsResearchQueue ? "Show all hypotheses" : "Show research queue")
            }
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    Task { await store.refresh() }
                } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .disabled(store.isLoading)
                .accessibilityLabel("Refresh opportunities")
            }
        }
        .refreshable {
            await store.refresh()
        }
        .task {
            await store.refreshIfNeeded()
            store.syncOpportunityHypotheses()
        }
        .sheet(item: $safariDestination) { destination in
            SafariView(url: destination.url)
        }
        .sheet(item: $selectedArticle) { article in
            ArticleDetailView(article: article)
        }
    }

    private var filteredHypotheses: [OpportunityHypothesis] {
        store.opportunityHypotheses.filter { hypothesis in
            (!showsResearchQueue || hypothesis.isSavedToResearchQueue)
                && (filter.label == nil || hypothesis.label == filter.label)
        }
    }

    private var labelCounts: [OpportunityLabel: Int] {
        Dictionary(grouping: store.opportunityHypotheses, by: \.label).mapValues(\.count)
    }

    private func open(_ article: ScoredArticle) {
        guard let url = URL(string: article.article.link) else { return }
        store.recordOpen(for: article)
        safariDestination = SafariDestination(url: url)
    }
}

private enum OpportunityInboxFilter: String, CaseIterable, Identifiable {
    case all
    case market
    case business
    case ai

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: "All"
        case .market: "Market"
        case .business: "Business"
        case .ai: "AI"
        }
    }

    var label: OpportunityLabel? {
        switch self {
        case .all: nil
        case .market: .marketMoving
        case .business: .businessOpportunity
        case .ai: .aiLaunch
        }
    }
}

private struct OpportunityInboxHeader: View {
    let totalCount: Int
    let savedCount: Int
    let counts: [OpportunityLabel: Int]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("\(totalCount) hypotheses", systemImage: "lightbulb.max")
                    .font(.headline)
                Spacer()
                Label("\(savedCount) saved", systemImage: "tray.full")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }

            Text("Hypotheses, not facts. Validate before committing product or research time.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                ForEach(OpportunityLabel.allCases) { label in
                    Label("\(counts[label, default: 0])", systemImage: label.systemImage)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
            }
        }
        .padding(14)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
    }
}

private struct OpportunityHypothesisCard: View {
    let hypothesis: OpportunityHypothesis
    let onToggleQueue: () -> Void
    let onViewEvidence: () -> Void
    let onOpenSource: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 5) {
                    Label("Hypothesis", systemImage: "lightbulb")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(.tint)
                    Text(hypothesis.supportingArticle.article.title)
                        .font(.headline)
                        .lineLimit(4)
                }
                Spacer(minLength: 8)
                VStack(alignment: .trailing, spacing: 5) {
                    Label(hypothesis.label.title, systemImage: hypothesis.label.systemImage)
                    Text(hypothesis.confidence.title)
                }
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
            }

            hypothesisField("Customer", hypothesis.customer)
            hypothesisField("Trigger", hypothesis.trigger)
            hypothesisField("Wedge", hypothesis.wedge)
            hypothesisField("Evidence", hypothesis.evidence)
            hypothesisField("Next test", hypothesis.nextTest)

            HStack(spacing: 12) {
                Button(action: onToggleQueue) {
                    Label(
                        hypothesis.isSavedToResearchQueue ? "Saved to research" : "Save to research",
                        systemImage: hypothesis.isSavedToResearchQueue ? "tray.full.fill" : "tray.and.arrow.down"
                    )
                    .frame(minHeight: 44)
                }
                .buttonStyle(.borderedProminent)

                Menu {
                    Button("View evidence", systemImage: "doc.text.magnifyingglass", action: onViewEvidence)
                    Button("Open source", systemImage: "safari", action: onOpenSource)
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .frame(minWidth: 44, minHeight: 44)
                }
                .accessibilityLabel("Hypothesis evidence actions")
            }
        }
        .padding(14)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
        .accessibilityElement(children: .contain)
    }

    private func hypothesisField(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

#Preview {
    NavigationStack {
        OpportunityInboxView()
            .environment(FounderModePreviewFixtures.freshFeed)
    }
}
