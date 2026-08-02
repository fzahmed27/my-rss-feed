import Foundation
import Observation

@MainActor
@Observable
final class FeedStore {
    var state: FeedLoadState = .idle
    var settings: FeedSettings {
        didSet {
            repository.saveSettings(settings)
        }
    }
    var presentationSettings: FeedPresentationSettings {
        didSet {
            repository.savePresentationSettings(presentationSettings)
            rerankCachedResult()
        }
    }
    var mutedTopics: [String: MutedTopic] {
        didSet {
            repository.saveMutedTopics(mutedTopics)
        }
    }
    var briefingSnapshots: [FounderBriefingSnapshot] {
        didSet {
            repository.saveBriefingSnapshots(briefingSnapshots)
        }
    }
    var briefingCompletions: [BriefingCompletion] {
        didSet {
            repository.saveBriefingCompletions(briefingCompletions)
        }
    }
    var opportunityHypotheses: [OpportunityHypothesis] {
        didSet {
            repository.saveOpportunityHypotheses(opportunityHypotheses)
        }
    }
    var founderProfile: FounderContextProfile {
        didSet {
            repository.saveFounderProfile(founderProfile)
            rerankCachedResult()
        }
    }
    var sources: [FeedSource] {
        didSet {
            repository.saveSources(sources, metrics: sourceMetrics)
        }
    }
    var interactions: [String: ArticleInteraction] {
        didSet {
            repository.saveInteractions(interactions)
        }
    }
    var savedArticles: [String: ScoredArticle] {
        didSet {
            repository.saveArticleSnapshots(savedArticles)
        }
    }
    var sourceMetrics: [String: SourceHealthSummary] {
        didSet {
            repository.saveSourceMetrics(sourceMetrics)
        }
    }

    private let client: FeedClient
    private let repository: FeedRepository

    convenience init(
        client: FeedClient = FeedClient(),
        sources overrideSources: [FeedSource]? = nil
    ) {
        self.init(
            client: client,
            repository: FeedRepository(),
            sources: overrideSources
        )
    }

    init(
        client: FeedClient,
        repository: FeedRepository,
        sources overrideSources: [FeedSource]? = nil
    ) {
        self.client = client
        self.repository = repository

        let repositorySettings = repository.loadSettings()
        let migratedSettings = Self.loadLegacySettings()
        let initialSettings = repositorySettings ?? migratedSettings
        let initialPresentationSettings = repository.loadPresentationSettings() ?? .defaults
        let initialMutedTopics = repository.loadMutedTopics()
        let initialBriefingSnapshots = repository.loadBriefingSnapshots()
        let initialBriefingCompletions = repository.loadBriefingCompletions()
        let initialOpportunityHypotheses = repository.loadOpportunityHypotheses()
        let initialFounderProfile = repository.loadFounderProfile() ?? .industrialAIFounder

        let repositorySources = repository.loadSources()
        let hasPersistedSources = repository.hasPersistedSources()
        let migratedSources = Self.loadLegacySources()
        let initialSources = overrideSources ?? (hasPersistedSources ? repositorySources : migratedSources)

        let repositoryMetrics = repository.loadSourceMetrics()

        let initialMetrics = repositoryMetrics

        let repositoryInteractions = repository.loadInteractions()
        let initialInteractions = repositoryInteractions.isEmpty ? Self.loadLegacyInteractions() : repositoryInteractions

        let repositorySnapshots = repository.loadArticleSnapshots()
        let initialSnapshots = repositorySnapshots.isEmpty ? Self.loadLegacySavedArticles() : repositorySnapshots

        settings = initialSettings
        presentationSettings = initialPresentationSettings
        mutedTopics = initialMutedTopics
        briefingSnapshots = initialBriefingSnapshots
        briefingCompletions = initialBriefingCompletions
        opportunityHypotheses = initialOpportunityHypotheses
        founderProfile = initialFounderProfile
        sources = initialSources
        sourceMetrics = initialMetrics
        interactions = initialInteractions
        savedArticles = initialSnapshots

        if repositorySettings == nil {
            repository.saveSettings(initialSettings)
        }

        if repository.loadPresentationSettings() == nil {
            repository.savePresentationSettings(initialPresentationSettings)
        }

        if repository.loadFounderProfile() == nil {
            repository.saveFounderProfile(initialFounderProfile)
        }

        if !hasPersistedSources {
            repository.saveSources(initialSources, metrics: initialMetrics)
        }

        if repositoryInteractions.isEmpty, !initialInteractions.isEmpty {
            repository.saveInteractions(initialInteractions)
        }

        if repositorySnapshots.isEmpty, !initialSnapshots.isEmpty {
            repository.saveArticleSnapshots(initialSnapshots)
        }

        if let cached = repository.loadCachedDigest(
            sources: sources,
            sourceMetrics: sourceMetrics,
            settings: settings
        ) {
            state = .loaded(cached)
        }

        syncOpportunityHypotheses()
    }

    var result: DigestResult? {
        state.result
    }

    var latestBriefingSnapshot: FounderBriefingSnapshot? {
        briefingSnapshots.first
    }

    var currentBriefingSnapshot: FounderBriefingSnapshot? {
        guard let result else { return nil }
        let items = FounderBriefingGenerator.generate(from: result.articles)
        return briefingSnapshots.first {
            $0.hasSameContent(
                generatedAt: result.generatedAt,
                intent: presentationSettings.selectedIntent,
                readingBudgetMinutes: presentationSettings.readingBudgetMinutes,
                items: items
            )
        }
    }

    var currentBriefingCompletion: BriefingCompletion? {
        guard let snapshot = currentBriefingSnapshot else { return nil }
        return completion(for: snapshot.id)
    }

    var savedResearchHypotheses: [OpportunityHypothesis] {
        opportunityHypotheses.filter(\.isSavedToResearchQueue)
    }

    var isLoading: Bool {
        if case .loading = state {
            return true
        }
        return false
    }

    var bookmarkCount: Int {
        interactions.values.filter(\.isBookmarked).count
    }

    var bookmarkedArticles: [ScoredArticle] {
        savedArticles.values
            .filter { interactions[$0.id]?.isBookmarked == true }
            .sorted { first, second in
                let firstDate = interactions[first.id]?.updatedAt ?? .distantPast
                let secondDate = interactions[second.id]?.updatedAt ?? .distantPast
                return firstDate > secondDate
            }
    }

    var opportunityArticles: [ScoredArticle] {
        var articlesByID = savedArticles.filter { !$0.value.opportunityLabels.isEmpty }

        for article in result?.articles ?? [] where !article.opportunityLabels.isEmpty {
            articlesByID[article.id] = article
        }

        return articlesByID.values.sorted { first, second in
            if first.score == second.score {
                return (first.article.publishedAt ?? .distantPast) > (second.article.publishedAt ?? .distantPast)
            }
            return first.score > second.score
        }
    }

    var feedbackCount: Int {
        interactions.values.filter { $0.feedback != nil }.count
    }

    var openedCount: Int {
        interactions.values.reduce(0) { $0 + $1.openedCount }
    }

    var activeSources: [FeedSource] {
        sources.filter(\.isActive)
    }

    func refreshIfNeeded() async {
        if case .idle = state {
            await refresh()
        }
    }

    func refresh() async {
        state = .loading

        let fetchSources = activeSources
        let learningProfile = makeLearningProfile()
        let fetchedResult = await client.fetchDigest(
            sources: fetchSources,
            settings: settings,
            intent: presentationSettings.selectedIntent,
            preferenceProfile: founderProfile.rankingPreferenceProfile,
            learningProfile: learningProfile
        )

        var healthBySourceID = Dictionary(uniqueKeysWithValues: fetchedResult.sourceHealth.map { ($0.source.id, $0) })

        for source in sources where !source.isActive {
            healthBySourceID[source.id] = SourceHealth(
                source: source,
                status: .paused,
                itemCount: 0,
                message: source.isEnabled ? "Muted from ranking and fetching." : "Disabled.",
                duration: 0
            )
        }

        let sourceHealth = sources.compactMap { healthBySourceID[$0.id] }
        updateMetrics(with: sourceHealth, rankedArticles: fetchedResult.articles)

        let result = DigestResult(
            articles: fetchedResult.articles,
            sourceHealth: sourceHealth.sorted { $0.source.name < $1.source.name },
            generatedAt: fetchedResult.generatedAt,
            fetchedCount: fetchedResult.fetchedCount
        )

        repository.saveRankedArticles(result.articles)
        state = .loaded(result)
        syncOpportunityHypotheses()
    }

    @discardableResult
    func saveCurrentBriefingSnapshot(savedAt: Date = Date()) -> FounderBriefingSnapshot? {
        guard let result else { return nil }
        if let existing = currentBriefingSnapshot {
            return existing
        }

        let snapshot = FounderBriefingSnapshot(
            generatedAt: result.generatedAt,
            savedAt: savedAt,
            intent: presentationSettings.selectedIntent,
            readingBudgetMinutes: presentationSettings.readingBudgetMinutes,
            fetchedCount: result.fetchedCount,
            items: FounderBriefingGenerator.generate(from: result.articles)
        )
        briefingSnapshots = Array(([snapshot] + briefingSnapshots).prefix(30))
        return snapshot
    }

    func recordBriefingShare(id: UUID, at date: Date = Date()) {
        guard let index = briefingSnapshots.firstIndex(where: { $0.id == id }) else { return }
        briefingSnapshots[index].shareCount += 1
        briefingSnapshots[index].lastSharedAt = date
    }

    func completion(for snapshotID: UUID) -> BriefingCompletion? {
        briefingCompletions.first { $0.snapshotID == snapshotID }
    }

    @discardableResult
    func completeCurrentBriefing(at date: Date = Date()) -> BriefingCompletion? {
        guard let snapshot = saveCurrentBriefingSnapshot() else { return nil }
        if let existing = completion(for: snapshot.id) {
            return existing
        }

        let completion = BriefingCompletion(
            snapshotID: snapshot.id,
            generatedAt: snapshot.generatedAt,
            completedAt: date,
            intent: snapshot.intent,
            readingBudgetMinutes: snapshot.readingBudgetMinutes,
            reviewedSlotCount: snapshot.items.count
        )
        briefingCompletions = Array(([completion] + briefingCompletions).prefix(90))
        return completion
    }

    func syncOpportunityHypotheses(from articles: [ScoredArticle]? = nil, at date: Date = Date()) {
        let generated = OpportunityHypothesisGenerator.generate(
            from: articles ?? opportunityArticles,
            createdAt: date
        )
        guard !generated.isEmpty else { return }

        let existingByID = Dictionary(uniqueKeysWithValues: opportunityHypotheses.map { ($0.id, $0) })
        let merged = generated.map { candidate in
            var candidate = candidate
            if let existing = existingByID[candidate.id] {
                candidate.isSavedToResearchQueue = existing.isSavedToResearchQueue
            }
            return candidate
        }
        let generatedIDs = Set(merged.map(\.id))
        let historical = opportunityHypotheses.filter { !generatedIDs.contains($0.id) }
        opportunityHypotheses = Array((merged + historical).prefix(200))
    }

    func toggleResearchQueue(id: String) {
        guard let index = opportunityHypotheses.firstIndex(where: { $0.id == id }) else { return }
        opportunityHypotheses[index].isSavedToResearchQueue.toggle()
    }

    func updateMinimumScore(_ value: Double) {
        settings.minimumScore = value
    }

    func updateDaysBack(_ value: Int) {
        settings.daysBack = value
    }

    func updateMaxArticles(_ value: Int) {
        settings.maxArticles = value
    }

    func updateFullTextExtractionEnabled(_ value: Bool) {
        settings.isFullTextExtractionEnabled = value
    }

    func updateFullTextArticleLimit(_ value: Int) {
        settings.fullTextArticleLimit = value
    }

    func resetSettings() {
        settings = .defaults
    }

    func updateFounderProfile(_ profile: FounderContextProfile) {
        founderProfile = profile
    }

    func resetFounderProfile() {
        founderProfile = .industrialAIFounder
    }

    func makeFounderProfileExportURL() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("signal-feed-founder-profile-\(Int(Date().timeIntervalSince1970))")
            .appendingPathExtension("txt")
        try Data(founderProfile.exportText.utf8).write(to: url, options: .atomic)
        return url
    }

    func updateIntent(_ intent: IntentMode) {
        presentationSettings.selectedIntent = intent
    }

    func updateReadingBudgetMinutes(_ minutes: Int) {
        presentationSettings.readingBudgetMinutes = max(5, min(60, minutes))
    }

    func interaction(for article: ScoredArticle) -> ArticleInteraction {
        interactions[article.id] ?? .empty
    }

    func isBookmarked(_ article: ScoredArticle) -> Bool {
        interaction(for: article).isBookmarked
    }

    func feedback(for article: ScoredArticle) -> ArticleFeedback? {
        interaction(for: article).feedback
    }

    func isRead(_ article: ScoredArticle) -> Bool {
        interaction(for: article).isRead
    }

    func isDismissed(_ article: ScoredArticle) -> Bool {
        interaction(for: article).isDismissed
    }

    func toggleBookmark(for article: ScoredArticle) {
        var interaction = interaction(for: article)
        interaction.isBookmarked.toggle()
        interaction.updatedAt = Date()
        updateInteraction(interaction, for: article)
    }

    func setFeedback(_ feedback: ArticleFeedback, for article: ScoredArticle) {
        setFeedback(feedback, reasons: [], for: article)
    }

    func setFeedback(_ feedback: ArticleFeedback, reasons: [FeedbackReason], for article: ScoredArticle) {
        var interaction = interaction(for: article)
        interaction.feedback = interaction.feedback == feedback ? nil : feedback
        interaction.feedbackReasons = interaction.feedback == nil ? [] : reasons
        interaction.updatedAt = Date()
        updateInteraction(interaction, for: article)
    }

    func clearFeedback(for article: ScoredArticle) {
        var interaction = interaction(for: article)
        interaction.feedback = nil
        interaction.feedbackReasons = []
        interaction.updatedAt = Date()
        updateInteraction(interaction, for: article)
    }

    func markRead(_ article: ScoredArticle, isRead: Bool = true) {
        var interaction = interaction(for: article)
        interaction.isRead = isRead
        interaction.updatedAt = Date()
        updateInteraction(interaction, for: article)
    }

    func dismiss(_ article: ScoredArticle) {
        var interaction = interaction(for: article)
        interaction.isDismissed = true
        interaction.isRead = true
        interaction.updatedAt = Date()
        updateInteraction(interaction, for: article)
    }

    func undoDismiss(_ article: ScoredArticle) {
        var interaction = interaction(for: article)
        interaction.isDismissed = false
        interaction.updatedAt = Date()
        updateInteraction(interaction, for: article)
    }

    func mutePrimaryTopic(for article: ScoredArticle, days: Int = 30) {
        guard let topic = article.matchedKeywords.first ?? article.category.title.lowercased().nilIfEmpty else { return }
        let key = topic.lowercased()
        mutedTopics[key] = MutedTopic(
            topic: topic,
            expiresAt: Calendar.current.date(byAdding: .day, value: days, to: Date()) ?? Date().addingTimeInterval(Double(days) * 86_400),
            createdAt: Date()
        )
    }

    func recordOpen(for article: ScoredArticle) {
        var interaction = interaction(for: article)
        interaction.openedCount += 1
        interaction.isRead = true
        interaction.lastOpenedAt = Date()
        interaction.updatedAt = Date()
        updateInteraction(interaction, for: article)
    }

    func addSource(
        name: String,
        url: String,
        kind: SourceKind,
        reputation: Double,
        isEnabled: Bool = true,
        isMuted: Bool = false
    ) {
        let source = FeedSource(
            id: uniqueSourceID(name: name, url: url),
            name: name.trimmingCharacters(in: .whitespacesAndNewlines),
            url: normalizedURL(url),
            reputation: reputation,
            kind: kind,
            isEnabled: isEnabled,
            isMuted: isEnabled && isMuted
        )
        sources.append(source)
    }

    @discardableResult
    func addDiscoveredFeed(_ feed: DiscoveredFeed) -> Bool {
        guard !sourceExists(url: feed.url) else { return false }
        addSource(
            name: feed.title,
            url: feed.url,
            kind: .feed,
            reputation: 1.0
        )
        return true
    }

    @discardableResult
    func importDiscoveredFeeds(_ feeds: [DiscoveredFeed]) -> Int {
        feeds.reduce(0) { count, feed in
            count + (addDiscoveredFeed(feed) ? 1 : 0)
        }
    }

    @discardableResult
    func importOPML(data: Data) throws -> Int {
        let feeds = try OPMLService.parse(data: data)
        return importDiscoveredFeeds(feeds)
    }

    func makeOPMLExportURL() throws -> URL {
        let opml = OPMLService.export(sources: sources)
        let data = Data(opml.utf8)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("signal-feed-sources-\(Int(Date().timeIntervalSince1970))")
            .appendingPathExtension("opml")
        try data.write(to: url, options: .atomic)
        return url
    }

    func makeFeedbackExportURL() throws -> URL {
        let lines = feedbackExportLines()
        let data = lines.joined(separator: "\n").data(using: .utf8) ?? Data()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("signal-feed-feedback-\(Int(Date().timeIntervalSince1970))")
            .appendingPathExtension("txt")
        try data.write(to: url, options: .atomic)
        return url
    }

    func updateSource(_ source: FeedSource) {
        guard let index = sources.firstIndex(where: { $0.id == source.id }) else { return }
        sources[index] = FeedSource(
            id: source.id,
            name: source.name.trimmingCharacters(in: .whitespacesAndNewlines),
            url: normalizedURL(source.url),
            reputation: source.reputation,
            kind: source.kind,
            isEnabled: source.isEnabled,
            isMuted: source.isMuted
        )
    }

    func deleteSource(_ source: FeedSource) {
        sources.removeAll { $0.id == source.id }
        sourceMetrics.removeValue(forKey: source.id)
    }

    func toggleSourceEnabled(_ source: FeedSource) {
        guard var updated = sources.first(where: { $0.id == source.id }) else { return }
        updated.isEnabled.toggle()
        if !updated.isEnabled {
            updated.isMuted = false
        }
        updateSource(updated)
    }

    func toggleSourceMuted(_ source: FeedSource) {
        guard var updated = sources.first(where: { $0.id == source.id }) else { return }
        updated.isMuted.toggle()
        if updated.isMuted {
            updated.isEnabled = true
        }
        updateSource(updated)
    }

    func resetSources() {
        sources = DefaultContentConfig.sources
        sourceMetrics = [:]
    }

    func metrics(for source: FeedSource) -> SourceHealthSummary {
        sourceMetrics[source.id] ?? .empty
    }

    private func updateInteraction(_ interaction: ArticleInteraction, for article: ScoredArticle) {
        if shouldKeep(interaction) {
            interactions[article.id] = interaction
            savedArticles[article.id] = article
        } else {
            interactions.removeValue(forKey: article.id)
            savedArticles.removeValue(forKey: article.id)
        }
    }

    private func shouldKeep(_ interaction: ArticleInteraction) -> Bool {
        interaction.isBookmarked || interaction.feedback != nil || interaction.openedCount > 0 || interaction.isRead || interaction.isDismissed
    }

    private func updateMetrics(with healthRows: [SourceHealth], rankedArticles: [ScoredArticle]) {
        var next = sourceMetrics
        let scoresBySource = Dictionary(grouping: rankedArticles, by: { $0.article.sourceID })
            .mapValues { articles in
                guard !articles.isEmpty else { return 0.0 }
                return articles.reduce(0.0) { $0 + $1.score } / Double(articles.count)
            }

        for health in healthRows {
            var summary = next[health.source.id] ?? .empty
            summary.lastStatus = health.status
            summary.lastMessage = health.message
            summary.lastItemCount = health.itemCount
            summary.lastDuration = health.duration

            if health.status != .paused {
                summary.lastFetchAt = Date()
                summary.totalFetches += 1
            }

            if health.status == .healthy || health.status == .empty {
                summary.lastSuccessfulFetchAt = Date()
                summary.failureStreak = 0
            } else if health.status == .failed {
                summary.failureStreak += 1
            }

            if let currentQuality = scoresBySource[health.source.id] {
                summary.averageQuality = summary.averageQuality == 0
                    ? currentQuality
                    : (summary.averageQuality * 0.75) + (currentQuality * 0.25)
            }

            next[health.source.id] = summary
        }

        sourceMetrics = next
    }

    private func makeLearningProfile() -> RankingLearningProfile {
        var sourceAdjustments: [String: Double] = [:]
        var keywordAdjustments: [String: Double] = [:]
        var topicAdjustments: [String: Double] = [:]

        for (articleID, interaction) in interactions {
            guard let article = savedArticles[articleID] else { continue }

            var weight = min(Double(interaction.openedCount) * 0.2, 1.0)
            if interaction.isBookmarked {
                weight += 1.0
            }

            switch interaction.feedback {
            case .liked:
                weight += 1.4
            case .disliked:
                weight -= 2.2
            case nil:
                break
            }

            guard weight != 0 else { continue }
            sourceAdjustments[article.article.sourceID, default: 0] += weight

            for keyword in article.matchedKeywords {
                let reasonAdjustment = interaction.feedbackReasons.reduce(0.0) { $0 + $1.adjustment }
                let adjustedWeight = weight + max(-0.8, min(0.8, reasonAdjustment))
                keywordAdjustments[keyword.lowercased(), default: 0] += adjustedWeight * 0.25
                topicAdjustments[keyword.lowercased(), default: 0] += adjustedWeight * 0.18
            }
        }

        let activeMutedTopics = mutedTopics
            .filter { $0.value.isActive() }
            .mapValues(\.expiresAt)

        return RankingLearningProfile(
            sourceAdjustments: sourceAdjustments.mapValues { max(-4, min(4, $0)) },
            keywordAdjustments: keywordAdjustments.mapValues { max(-2, min(2, $0)) },
            topicAdjustments: topicAdjustments.mapValues { max(-1.5, min(1.5, $0)) },
            mutedTopics: activeMutedTopics
        )
    }

    private func rerankCachedResult() {
        guard case .loaded(let result) = state else { return }
        let ranked = RankingEngine.rank(
            articles: result.articles.map(\.article),
            settings: settings,
            intent: presentationSettings.selectedIntent,
            preferenceProfile: founderProfile.rankingPreferenceProfile,
            learningProfile: makeLearningProfile()
        )
        let updated = DigestResult(
            articles: ranked,
            sourceHealth: result.sourceHealth,
            generatedAt: result.generatedAt,
            fetchedCount: result.fetchedCount
        )
        repository.saveRankedArticles(updated.articles)
        state = .loaded(updated)
        syncOpportunityHypotheses(from: updated.articles)
    }

    private func feedbackExportLines() -> [String] {
        var lines: [String] = [
            "Signal Feed feedback export",
            "Generated: \(ISO8601DateFormatter().string(from: Date()))",
            "Intent mode: \(presentationSettings.selectedIntent.title)",
            "",
            "Muted topics"
        ]

        if mutedTopics.isEmpty {
            lines.append("- None")
        } else {
            for topic in mutedTopics.values.sorted(by: { $0.topic < $1.topic }) {
                lines.append("- \(topic.topic), expires \(ISO8601DateFormatter().string(from: topic.expiresAt))")
            }
        }

        lines.append("")
        lines.append("Article interactions")

        let sortedInteractions = interactions.sorted { first, second in
            first.value.updatedAt > second.value.updatedAt
        }

        if sortedInteractions.isEmpty {
            lines.append("- None")
        } else {
            for (articleID, interaction) in sortedInteractions {
                let article = savedArticles[articleID]
                let reasons = interaction.feedbackReasons.map(\.title).joined(separator: ", ")
                let title = article?.article.title ?? articleID
                lines.append("- \(title)")
                lines.append("  feedback: \(interaction.feedback?.rawValue ?? "none")")
                lines.append("  reasons: \(reasons.isEmpty ? "none" : reasons)")
                lines.append("  bookmarked: \(interaction.isBookmarked)")
                lines.append("  read: \(interaction.isRead)")
                lines.append("  dismissed: \(interaction.isDismissed)")
                lines.append("  opens: \(interaction.openedCount)")
                lines.append("  updated: \(ISO8601DateFormatter().string(from: interaction.updatedAt))")
            }
        }

        return lines
    }

    private func uniqueSourceID(name: String, url: String) -> String {
        let rawBase = slug(from: name.isEmpty ? url : name)
        let base = rawBase.isEmpty ? "source" : rawBase
        var candidate = base
        var suffix = 2

        while sources.contains(where: { $0.id == candidate }) {
            candidate = "\(base)-\(suffix)"
            suffix += 1
        }

        return candidate
    }

    private func slug(from value: String) -> String {
        value
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .joined(separator: "-")
    }

    private func normalizedURL(_ rawValue: String) -> String {
        rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func sourceExists(url: String) -> Bool {
        let normalized = normalizedURL(url).lowercased()
        return sources.contains { normalizedURL($0.url).lowercased() == normalized }
    }
}

private extension String {
    var nilIfEmpty: String? {
        isEmpty ? nil : self
    }
}

private extension FeedStore {
    static func loadLegacySettings() -> FeedSettings {
        guard let data = UserDefaults.standard.data(forKey: "signalFeed.settings"),
              let settings = try? JSONDecoder().decode(FeedSettings.self, from: data) else {
            return .defaults
        }
        return settings
    }

    static func loadLegacySources() -> [FeedSource] {
        guard let data = UserDefaults.standard.data(forKey: "signalFeed.sources"),
              let sources = try? JSONDecoder().decode([FeedSource].self, from: data),
              !sources.isEmpty else {
            return DefaultContentConfig.sources
        }
        return sources
    }

    static func loadLegacyInteractions() -> [String: ArticleInteraction] {
        guard let data = UserDefaults.standard.data(forKey: "signalFeed.interactions"),
              let interactions = try? JSONDecoder().decode([String: ArticleInteraction].self, from: data) else {
            return [:]
        }
        return interactions
    }

    static func loadLegacySavedArticles() -> [String: ScoredArticle] {
        guard let data = UserDefaults.standard.data(forKey: "signalFeed.savedArticles"),
              let articles = try? JSONDecoder().decode([String: ScoredArticle].self, from: data) else {
            return [:]
        }
        return articles
    }
}

extension FeedStore {
    static var preview: FeedStore {
        FounderModePreviewFixtures.freshFeed
    }
}
