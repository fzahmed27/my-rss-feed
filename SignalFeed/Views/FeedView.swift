import SwiftUI

struct FeedView: View {
    @Environment(FeedStore.self) private var store
    @State private var searchQuery = ""
    @State private var selectedCategory: ArticleCategory?
    @State private var showsBookmarksOnly = false
    @State private var showsFullFeed = false
    @State private var showsMore = false
    @State private var safariDestination: SafariDestination?
    @State private var selectedArticle: ScoredArticle?

    var body: some View {
        content
            .navigationTitle("Signal Feed")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showsBookmarksOnly.toggle()
                        selectedCategory = nil
                    } label: {
                        Image(systemName: showsBookmarksOnly ? "bookmark.fill" : "bookmark")
                    }
                    .accessibilityLabel(showsBookmarksOnly ? "Show all articles" : "Show saved articles")
                }

                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        Task { await store.refresh() }
                    } label: {
                        Image(systemName: "arrow.clockwise")
                    }
                    .disabled(store.isLoading)
                    .accessibilityLabel("Refresh feeds")
                }
            }
            .searchable(
                text: $searchQuery,
                placement: .navigationBarDrawer(displayMode: .always),
                prompt: "Search titles, sources, reasons"
            )
            .refreshable {
                await store.refresh()
            }
            .task {
                await store.refreshIfNeeded()
            }
            .sheet(item: $safariDestination) { destination in
                SafariView(url: destination.url)
            }
            .sheet(item: $selectedArticle) { article in
                ArticleDetailView(article: article)
            }
    }

    @ViewBuilder
    private var content: some View {
        switch store.state {
        case .idle, .loading:
            loadingList
        case .loaded(let result):
            articleList(result: result)
        case .failed(let message):
            ContentUnavailableView(
                "Could not load feeds",
                systemImage: "wifi.exclamationmark",
                description: Text(message)
            )
        }
    }

    private var loadingList: some View {
        List {
            ForEach(0..<5, id: \.self) { _ in
                    ArticleCard(
                        article: .placeholder,
                        isBookmarked: false,
                    interaction: .empty,
                        onSelect: {},
                        onOpen: {},
                        onToggleBookmark: {},
                    onFeedback: { _, _ in },
                    onClearFeedback: {},
                    onToggleRead: {},
                    onDismiss: {},
                    onMuteTopic: {}
                )
                    .redacted(reason: .placeholder)
                    .allowsHitTesting(false)
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                    .listRowSeparator(.hidden)
            }
        }
        .listStyle(.plain)
        .environment(\.defaultMinListRowHeight, 1)
    }

    private func articleList(result: DigestResult) -> some View {
        let baseArticles = showsBookmarksOnly ? store.bookmarkedArticles : result.articles
        let articles = filteredArticles(from: baseArticles)

        return List {
            Section {
                SummaryHeader(result: result)
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                    .listRowSeparator(.hidden)

                CategoryFilterBar(
                    counts: categoryCounts(from: baseArticles),
                    selection: $selectedCategory
                )
                .listRowInsets(EdgeInsets(top: 0, leading: 16, bottom: 8, trailing: 16))
                .listRowSeparator(.hidden)

                IntentSelector(
                    selection: Binding(
                        get: { store.presentationSettings.selectedIntent },
                        set: { store.updateIntent($0) }
                    )
                )
                .listRowInsets(EdgeInsets(top: 0, leading: 16, bottom: 8, trailing: 16))
                .listRowSeparator(.hidden)

                if let mutedTopicSummary {
                    Label(mutedTopicSummary, systemImage: "speaker.slash.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .accessibilityLabel(mutedTopicSummary)
                        .listRowInsets(EdgeInsets(top: 0, leading: 16, bottom: 8, trailing: 16))
                        .listRowSeparator(.hidden)
                }
            }

            if articles.isEmpty {
                ContentUnavailableView(
                    emptyStateTitle(result: result),
                    systemImage: "line.3.horizontal.decrease.circle",
                    description: Text(emptyStateDescription(result: result))
                )
                .listRowSeparator(.hidden)
            } else {
                let budget = ReadingBudget(articles: articles, interactions: store.interactions)

                if showsBookmarksOnly || showsFullFeed {
                    Section {
                        ForEach(articles) { article in
                            articleRow(article)
                        }
                    } header: {
                        Label(showsBookmarksOnly ? "Saved" : "Full Feed", systemImage: showsBookmarksOnly ? "bookmark.fill" : "tray.full")
                    }
                } else {
                    Section {
                        HStack {
                            Label("\(budget.estimatedReviewMinutes) min planned", systemImage: "timer")
                            Spacer()
                            Text("\(store.presentationSettings.readingBudgetMinutes) min budget")
                                .foregroundStyle(.secondary)
                        }
                        .font(.caption.weight(.semibold))
                    }

                    Section {
                        ProgressHeader(title: "Must Read", readCount: budget.mustReadReadCount, totalCount: budget.mustRead.count)
                        ForEach(budget.mustRead) { article in
                            articleRow(article)
                        }
                    } header: {
                        Label("Must Read", systemImage: "exclamationmark.circle")
                    }

                    Section {
                        ProgressHeader(title: "Worth Skimming", readCount: budget.worthSkimmingReadCount, totalCount: budget.worthSkimming.count)
                        ForEach(budget.worthSkimming) { article in
                            articleRow(article)
                        }
                        if budget.isCaughtUp {
                            Text("You're caught up. Go build.")
                                .font(.headline)
                                .foregroundStyle(.secondary)
                                .frame(maxWidth: .infinity, alignment: .center)
                                .padding(.vertical, 12)
                        }
                    } header: {
                        Label("Worth Skimming", systemImage: "text.line.first.and.arrowtriangle.forward")
                    }

                    if !budget.more.isEmpty {
                        Section {
                            DisclosureGroup(isExpanded: $showsMore) {
                                ForEach(budget.more) { article in
                                    articleRow(article)
                                }
                            } label: {
                                HStack {
                                    Label("More", systemImage: "tray.full")
                                    Spacer()
                                    Text("\(budget.more.count)")
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .padding(.vertical, 6)
                        } footer: {
                            Button {
                                showsFullFeed = true
                            } label: {
                                Label("Show full feed", systemImage: "list.bullet")
                            }
                        }
                    }
                }
            }
        }
        .listStyle(.plain)
        .environment(\.defaultMinListRowHeight, 1)
    }

    private func articleRow(_ article: ScoredArticle) -> some View {
        ArticleCard(
            article: article,
            isBookmarked: store.isBookmarked(article),
            interaction: store.interaction(for: article),
            onSelect: {
                selectedArticle = article
            },
            onOpen: {
                open(article)
            },
            onToggleBookmark: {
                store.toggleBookmark(for: article)
            },
            onFeedback: { feedback, reasons in
                store.setFeedback(feedback, reasons: reasons, for: article)
            },
            onClearFeedback: {
                store.clearFeedback(for: article)
            },
            onToggleRead: {
                store.markRead(article, isRead: !store.isRead(article))
            },
            onDismiss: {
                store.dismiss(article)
            },
            onMuteTopic: {
                store.mutePrimaryTopic(for: article)
                Task { await store.refresh() }
            }
        )
        .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
        .listRowSeparator(.hidden)
    }

    private func filteredArticles(from articles: [ScoredArticle]) -> [ScoredArticle] {
        var articles = articles
        if let selectedCategory {
            articles = articles.filter { $0.category == selectedCategory }
        }

        let query = searchQuery
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()

        if !query.isEmpty {
            articles = articles.filter { article in
                article.article.searchableText.contains(query)
                    || article.reasons.joined(separator: " ").lowercased().contains(query)
                    || article.matchedKeywords.joined(separator: " ").lowercased().contains(query)
            }
        }

        return articles.filter { !store.isDismissed($0) }
    }

    private func categoryCounts(from articles: [ScoredArticle]) -> [ArticleCategory: Int] {
        Dictionary(grouping: articles, by: \.category).mapValues(\.count)
    }

    private var mutedTopicSummary: String? {
        let topics = store.mutedTopics.values
            .sorted { $0.topic.localizedCaseInsensitiveCompare($1.topic) == .orderedAscending }
        guard let first = topics.first else { return nil }
        let suffix = topics.count == 1 ? "" : " +\(topics.count - 1) more"
        return "\(first.topic) muted until \(first.expiresAt.formatted(date: .abbreviated, time: .omitted))\(suffix)"
    }

    private func emptyStateTitle(result: DigestResult) -> String {
        if showsBookmarksOnly { return "No saved items" }
        if searchQuery.isEmpty, selectedCategory == nil, result.fetchedCount > 0 {
            return "No high-quality articles"
        }
        return "No matching items"
    }

    private func emptyStateDescription(result: DigestResult) -> String {
        if showsBookmarksOnly {
            return "Save articles from the feed or briefing to keep them here."
        }
        if searchQuery.isEmpty, selectedCategory == nil, result.fetchedCount > 0 {
            return "\(result.fetchedCount) items were fetched, but none met the current quality threshold."
        }
        return "Try a lower score threshold, a broader date range, or a different search."
    }

    private func open(_ article: ScoredArticle) {
        guard let url = URL(string: article.article.link) else { return }
        store.recordOpen(for: article)
        safariDestination = SafariDestination(url: url)
    }
}

struct ReadingBudget {
    let mustRead: [ScoredArticle]
    let worthSkimming: [ScoredArticle]
    let more: [ScoredArticle]
    private let interactions: [String: ArticleInteraction]

    init(articles: [ScoredArticle], interactions: [String: ArticleInteraction]) {
        self.mustRead = Array(articles.prefix(5))
        self.worthSkimming = Array(articles.dropFirst(5).prefix(10))
        self.more = Array(articles.dropFirst(15))
        self.interactions = interactions
    }

    var mustReadReadCount: Int {
        mustRead.filter { interactions[$0.id]?.isRead == true }.count
    }

    var worthSkimmingReadCount: Int {
        worthSkimming.filter { interactions[$0.id]?.isRead == true }.count
    }

    var isCaughtUp: Bool {
        !worthSkimming.isEmpty && worthSkimmingReadCount == worthSkimming.count
    }

    var estimatedReviewMinutes: Int {
        mustRead.reduce(0) { $0 + $1.decisionSummary.estimatedReadingMinutes }
            + worthSkimming.reduce(0) { $0 + max(1, min(2, $1.decisionSummary.estimatedReadingMinutes)) }
    }
}

private struct ProgressHeader: View {
    let title: String
    let readCount: Int
    let totalCount: Int

    var body: some View {
        HStack {
            Text("\(readCount) of \(totalCount) \(title)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Spacer()
            ProgressView(value: Double(readCount), total: Double(max(totalCount, 1)))
                .frame(width: 92)
        }
        .padding(.vertical, 4)
    }
}

private struct IntentSelector: View {
    @Binding var selection: IntentMode

    var body: some View {
        Picker("Intent", selection: $selection) {
            ForEach(IntentMode.allCases) { mode in
                Text(mode.title).tag(mode)
            }
        }
        .pickerStyle(.segmented)
        .accessibilityLabel("Feed intent")
    }
}

private struct SummaryHeader: View {
    let result: DigestResult

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("\(result.articles.count) ranked", systemImage: "sparkline")
                Spacer()
                Text(result.generatedAt.formatted(date: .omitted, time: .shortened))
            }
            .font(.subheadline.weight(.semibold))

            HStack(spacing: 12) {
                Label("\(result.fetchedCount) fetched", systemImage: "tray.and.arrow.down")
                Label("\(healthySources) healthy", systemImage: "checkmark.circle")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
        .padding(14)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 8))
    }

    private var healthySources: Int {
        result.sourceHealth.filter { $0.status == .healthy }.count
    }
}

private struct CategoryFilterBar: View {
    let counts: [ArticleCategory: Int]
    @Binding var selection: ArticleCategory?

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                FilterChip(
                    title: "All",
                    systemImage: "square.grid.2x2",
                    count: counts.values.reduce(0, +),
                    isSelected: selection == nil
                ) {
                    selection = nil
                }

                ForEach(ArticleCategory.allCases.filter { (counts[$0] ?? 0) > 0 }) { category in
                    FilterChip(
                        title: category.title,
                        systemImage: category.systemImage,
                        count: counts[category, default: 0],
                        isSelected: selection == category
                    ) {
                        selection = category
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }
}

private struct FilterChip: View {
    let title: String
    let systemImage: String
    let count: Int
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label("\(title) \(count)", systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .lineLimit(1)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .foregroundStyle(isSelected ? .white : .primary)
                .background(
                    isSelected ? Color.accentColor : Color(.secondarySystemBackground),
                    in: Capsule()
                )
        }
        .buttonStyle(.plain)
    }
}

struct SafariDestination: Identifiable {
    let url: URL
    var id: URL { url }
}

#Preview("Fresh feed") {
    NavigationStack {
        FeedView()
            .environment(FounderModePreviewFixtures.freshFeed)
    }
}

#Preview("Fully caught up") {
    NavigationStack {
        FeedView()
            .environment(FounderModePreviewFixtures.caughtUpFeed)
    }
}

#Preview("No high-quality articles") {
    NavigationStack {
        FeedView()
            .environment(FounderModePreviewFixtures.noHighQualityFeed)
    }
}

#Preview("Muted topic") {
    NavigationStack {
        FeedView()
            .environment(FounderModePreviewFixtures.mutedTopicFeed)
    }
}

#Preview("Accessibility text") {
    NavigationStack {
        FeedView()
            .environment(FounderModePreviewFixtures.freshFeed)
            .environment(\.dynamicTypeSize, .accessibility3)
    }
}
