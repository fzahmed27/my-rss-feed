import Foundation
import SwiftData

@MainActor
final class FeedRepository {
    private enum BlobKey {
        static let settings = "settings"
        static let presentationSettings = "presentationSettings"
        static let mutedTopics = "mutedTopics"
        static let briefingSnapshots = "briefingSnapshots"
        static let briefingCompletions = "briefingCompletions"
        static let opportunityHypotheses = "opportunityHypotheses"
        static let founderProfile = "founderProfile"
        static let sourcesInitialized = "sourcesInitialized"
    }

    private let context: ModelContext

    init(inMemory: Bool = false) {
        let schema = Schema([
            PersistentBlob.self,
            PersistentSourceRecord.self,
            PersistentInteractionRecord.self,
            PersistentArticleSnapshotRecord.self,
            PersistentRankedArticleRecord.self
        ])
        let configuration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: inMemory)

        do {
            let container = try ModelContainer(for: schema, configurations: [configuration])
            context = ModelContext(container)
        } catch {
            let fallbackConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: true)
            let fallbackContainer = try! ModelContainer(for: schema, configurations: [fallbackConfiguration])
            context = ModelContext(fallbackContainer)
        }
    }

    func loadSettings() -> FeedSettings? {
        guard let blob = blob(for: BlobKey.settings) else { return nil }
        return try? JSONDecoder().decode(FeedSettings.self, from: blob.data)
    }

    func saveSettings(_ settings: FeedSettings) {
        guard let data = try? JSONEncoder().encode(settings) else { return }
        upsertBlob(key: BlobKey.settings, data: data)
    }

    func loadPresentationSettings() -> FeedPresentationSettings? {
        guard let blob = blob(for: BlobKey.presentationSettings) else { return nil }
        return try? JSONDecoder().decode(FeedPresentationSettings.self, from: blob.data)
    }

    func savePresentationSettings(_ settings: FeedPresentationSettings) {
        guard let data = try? JSONEncoder().encode(settings) else { return }
        upsertBlob(key: BlobKey.presentationSettings, data: data)
    }

    func loadMutedTopics() -> [String: MutedTopic] {
        guard let blob = blob(for: BlobKey.mutedTopics),
              let topics = try? JSONDecoder().decode([String: MutedTopic].self, from: blob.data) else {
            return [:]
        }
        return topics.filter { $0.value.isActive() }
    }

    func saveMutedTopics(_ topics: [String: MutedTopic]) {
        let activeTopics = topics.filter { $0.value.isActive() }
        guard let data = try? JSONEncoder().encode(activeTopics) else { return }
        upsertBlob(key: BlobKey.mutedTopics, data: data)
    }

    func loadBriefingSnapshots() -> [FounderBriefingSnapshot] {
        guard let blob = blob(for: BlobKey.briefingSnapshots),
              let snapshots = try? JSONDecoder().decode([FounderBriefingSnapshot].self, from: blob.data) else {
            return []
        }
        return snapshots.sorted { $0.savedAt > $1.savedAt }
    }

    func saveBriefingSnapshots(_ snapshots: [FounderBriefingSnapshot]) {
        guard let data = try? JSONEncoder().encode(snapshots) else { return }
        upsertBlob(key: BlobKey.briefingSnapshots, data: data)
    }

    func loadBriefingCompletions() -> [BriefingCompletion] {
        guard let blob = blob(for: BlobKey.briefingCompletions),
              let completions = try? JSONDecoder().decode([BriefingCompletion].self, from: blob.data) else {
            return []
        }
        return completions.sorted { $0.completedAt > $1.completedAt }
    }

    func saveBriefingCompletions(_ completions: [BriefingCompletion]) {
        guard let data = try? JSONEncoder().encode(completions) else { return }
        upsertBlob(key: BlobKey.briefingCompletions, data: data)
    }

    func loadOpportunityHypotheses() -> [OpportunityHypothesis] {
        guard let blob = blob(for: BlobKey.opportunityHypotheses),
              let hypotheses = try? JSONDecoder().decode([OpportunityHypothesis].self, from: blob.data) else {
            return []
        }
        return hypotheses.sorted { $0.createdAt > $1.createdAt }
    }

    func saveOpportunityHypotheses(_ hypotheses: [OpportunityHypothesis]) {
        guard let data = try? JSONEncoder().encode(hypotheses) else { return }
        upsertBlob(key: BlobKey.opportunityHypotheses, data: data)
    }

    func loadFounderProfile() -> FounderContextProfile? {
        guard let blob = blob(for: BlobKey.founderProfile) else { return nil }
        return try? JSONDecoder().decode(FounderContextProfile.self, from: blob.data)
    }

    func saveFounderProfile(_ profile: FounderContextProfile) {
        guard let data = try? JSONEncoder().encode(profile) else { return }
        upsertBlob(key: BlobKey.founderProfile, data: data)
    }

    func loadSources() -> [FeedSource] {
        fetchSourceRecords()
            .sorted { $0.name < $1.name }
            .map(\.source)
    }

    func hasPersistedSources() -> Bool {
        blob(for: BlobKey.sourcesInitialized) != nil || !fetchSourceRecords().isEmpty
    }

    func saveSources(_ sources: [FeedSource], metrics: [String: SourceHealthSummary]) {
        let existing = Dictionary(uniqueKeysWithValues: fetchSourceRecords().map { ($0.id, $0) })
        let incomingIDs = Set(sources.map(\.id))

        for source in sources {
            let record = existing[source.id] ?? PersistentSourceRecord(source: source)
            record.apply(source: source)
            if let summary = metrics[source.id] {
                record.apply(summary: summary)
            }
            if existing[source.id] == nil {
                context.insert(record)
            }
        }

        for record in existing.values where !incomingIDs.contains(record.id) {
            context.delete(record)
        }

        saveContext()
        upsertBlob(key: BlobKey.sourcesInitialized, data: Data([1]))
    }

    func loadSourceMetrics() -> [String: SourceHealthSummary] {
        Dictionary(uniqueKeysWithValues: fetchSourceRecords().map { ($0.id, $0.summary) })
    }

    func saveSourceMetrics(_ metrics: [String: SourceHealthSummary]) {
        let records = fetchSourceRecords()
        for record in records {
            record.apply(summary: metrics[record.id] ?? .empty)
        }
        saveContext()
    }

    func loadInteractions() -> [String: ArticleInteraction] {
        Dictionary(uniqueKeysWithValues: fetchInteractionRecords().map { ($0.articleID, $0.interaction) })
    }

    func saveInteractions(_ interactions: [String: ArticleInteraction]) {
        let existing = Dictionary(uniqueKeysWithValues: fetchInteractionRecords().map { ($0.articleID, $0) })
        let incomingIDs = Set(interactions.keys)

        for (articleID, interaction) in interactions {
            let record = existing[articleID] ?? PersistentInteractionRecord(articleID: articleID, interaction: interaction)
            record.apply(interaction: interaction)
            if existing[articleID] == nil {
                context.insert(record)
            }
        }

        for record in existing.values where !incomingIDs.contains(record.articleID) {
            context.delete(record)
        }

        saveContext()
    }

    func loadArticleSnapshots() -> [String: ScoredArticle] {
        var snapshots: [String: ScoredArticle] = [:]
        for record in fetchArticleSnapshotRecords() {
            if let article = try? JSONDecoder().decode(ScoredArticle.self, from: record.payload) {
                snapshots[record.id] = article
            }
        }
        return snapshots
    }

    func saveArticleSnapshots(_ articles: [String: ScoredArticle]) {
        let existing = Dictionary(uniqueKeysWithValues: fetchArticleSnapshotRecords().map { ($0.id, $0) })
        let incomingIDs = Set(articles.keys)

        for (id, article) in articles {
            guard let payload = try? JSONEncoder().encode(article) else { continue }
            let record = existing[id] ?? PersistentArticleSnapshotRecord(id: id, payload: payload, updatedAt: Date())
            record.payload = payload
            record.updatedAt = Date()
            if existing[id] == nil {
                context.insert(record)
            }
        }

        for record in existing.values where !incomingIDs.contains(record.id) {
            context.delete(record)
        }

        saveContext()
    }

    func saveRankedArticles(_ articles: [ScoredArticle]) {
        let existing = Dictionary(uniqueKeysWithValues: fetchRankedArticleRecords().map { ($0.id, $0) })
        let incomingIDs = Set(articles.map(\.id))
        let refreshedAt = Date()

        for article in articles {
            guard let payload = try? JSONEncoder().encode(article) else { continue }
            let record = existing[article.id] ?? PersistentRankedArticleRecord(article: article, payload: payload, refreshedAt: refreshedAt)
            record.apply(article: article, payload: payload, refreshedAt: refreshedAt)
            if existing[article.id] == nil {
                context.insert(record)
            }
        }

        for record in existing.values where !incomingIDs.contains(record.id) {
            context.delete(record)
        }

        saveContext()
    }

    func loadCachedDigest(
        sources: [FeedSource],
        sourceMetrics: [String: SourceHealthSummary],
        settings: FeedSettings
    ) -> DigestResult? {
        let articles = fetchRankedArticleRecords()
            .compactMap { try? JSONDecoder().decode(ScoredArticle.self, from: $0.payload) }
            .filter { article in
                guard let publishedAt = article.article.publishedAt else { return true }
                guard let cutoff = Calendar.current.date(byAdding: .day, value: -settings.daysBack, to: Date()) else { return true }
                return publishedAt >= cutoff
            }
            .filter { $0.score >= settings.minimumScore }
            .sorted { $0.score > $1.score }

        guard !articles.isEmpty else { return nil }

        let health = sources.map { source in
            let summary = sourceMetrics[source.id] ?? .empty
            return SourceHealth(
                source: source,
                status: summary.lastStatus ?? (source.isActive ? .empty : .paused),
                itemCount: summary.lastItemCount,
                message: summary.lastMessage.isEmpty ? "Loaded from on-device storage." : summary.lastMessage,
                duration: summary.lastDuration
            )
        }

        let generatedAt = fetchRankedArticleRecords()
            .map(\.refreshedAt)
            .max() ?? Date()

        return DigestResult(
            articles: Array(articles.prefix(settings.maxArticles)),
            sourceHealth: health,
            generatedAt: generatedAt,
            fetchedCount: articles.count
        )
    }

    private func blob(for key: String) -> PersistentBlob? {
        fetchBlobs().first { $0.key == key }
    }

    private func upsertBlob(key: String, data: Data) {
        if let blob = blob(for: key) {
            blob.data = data
            blob.updatedAt = Date()
        } else {
            context.insert(PersistentBlob(key: key, data: data))
        }
        saveContext()
    }

    private func fetchBlobs() -> [PersistentBlob] {
        (try? context.fetch(FetchDescriptor<PersistentBlob>())) ?? []
    }

    private func fetchSourceRecords() -> [PersistentSourceRecord] {
        (try? context.fetch(FetchDescriptor<PersistentSourceRecord>())) ?? []
    }

    private func fetchInteractionRecords() -> [PersistentInteractionRecord] {
        (try? context.fetch(FetchDescriptor<PersistentInteractionRecord>())) ?? []
    }

    private func fetchArticleSnapshotRecords() -> [PersistentArticleSnapshotRecord] {
        (try? context.fetch(FetchDescriptor<PersistentArticleSnapshotRecord>())) ?? []
    }

    private func fetchRankedArticleRecords() -> [PersistentRankedArticleRecord] {
        (try? context.fetch(FetchDescriptor<PersistentRankedArticleRecord>())) ?? []
    }

    private func saveContext() {
        try? context.save()
    }
}

@Model
final class PersistentBlob {
    @Attribute(.unique) var key: String
    var data: Data
    var updatedAt: Date

    init(key: String, data: Data, updatedAt: Date = Date()) {
        self.key = key
        self.data = data
        self.updatedAt = updatedAt
    }
}

@Model
final class PersistentSourceRecord {
    @Attribute(.unique) var id: String
    var name: String
    var url: String
    var reputation: Double
    var kindRaw: String
    var isEnabled: Bool
    var isMuted: Bool
    var lastFetchAt: Date?
    var lastSuccessfulFetchAt: Date?
    var failureStreak: Int
    var totalFetches: Int
    var averageQuality: Double
    var lastStatusRaw: String?
    var lastMessage: String
    var lastItemCount: Int
    var lastDuration: TimeInterval
    var updatedAt: Date

    init(source: FeedSource) {
        id = source.id
        name = source.name
        url = source.url
        reputation = source.reputation
        kindRaw = source.kind.rawValue
        isEnabled = source.isEnabled
        isMuted = source.isMuted
        lastFetchAt = nil
        lastSuccessfulFetchAt = nil
        failureStreak = 0
        totalFetches = 0
        averageQuality = 0
        lastStatusRaw = nil
        lastMessage = ""
        lastItemCount = 0
        lastDuration = 0
        updatedAt = Date()
    }

    var source: FeedSource {
        FeedSource(
            id: id,
            name: name,
            url: url,
            reputation: reputation,
            kind: SourceKind(rawValue: kindRaw) ?? .feed,
            isEnabled: isEnabled,
            isMuted: isMuted
        )
    }

    var summary: SourceHealthSummary {
        SourceHealthSummary(
            lastFetchAt: lastFetchAt,
            lastSuccessfulFetchAt: lastSuccessfulFetchAt,
            failureStreak: failureStreak,
            totalFetches: totalFetches,
            averageQuality: averageQuality,
            lastStatus: lastStatusRaw.flatMap(SourceStatus.init(rawValue:)),
            lastMessage: lastMessage,
            lastItemCount: lastItemCount,
            lastDuration: lastDuration
        )
    }

    func apply(source: FeedSource) {
        name = source.name
        url = source.url
        reputation = source.reputation
        kindRaw = source.kind.rawValue
        isEnabled = source.isEnabled
        isMuted = source.isMuted
        updatedAt = Date()
    }

    func apply(summary: SourceHealthSummary) {
        lastFetchAt = summary.lastFetchAt
        lastSuccessfulFetchAt = summary.lastSuccessfulFetchAt
        failureStreak = summary.failureStreak
        totalFetches = summary.totalFetches
        averageQuality = summary.averageQuality
        lastStatusRaw = summary.lastStatus?.rawValue
        lastMessage = summary.lastMessage
        lastItemCount = summary.lastItemCount
        lastDuration = summary.lastDuration
        updatedAt = Date()
    }
}

@Model
final class PersistentInteractionRecord {
    @Attribute(.unique) var articleID: String
    var isBookmarked: Bool
    var feedbackRaw: String?
    var openedCount: Int
    var lastOpenedAt: Date?
    var updatedAt: Date
    var payload: Data?

    init(articleID: String, interaction: ArticleInteraction) {
        self.articleID = articleID
        isBookmarked = interaction.isBookmarked
        feedbackRaw = interaction.feedback?.rawValue
        openedCount = interaction.openedCount
        lastOpenedAt = interaction.lastOpenedAt
        updatedAt = interaction.updatedAt
        payload = try? JSONEncoder().encode(interaction)
    }

    var interaction: ArticleInteraction {
        if let payload,
           let interaction = try? JSONDecoder().decode(ArticleInteraction.self, from: payload) {
            return interaction
        }

        return ArticleInteraction(
            isBookmarked: isBookmarked,
            feedback: feedbackRaw.flatMap(ArticleFeedback.init(rawValue:)),
            openedCount: openedCount,
            lastOpenedAt: lastOpenedAt,
            updatedAt: updatedAt
        )
    }

    func apply(interaction: ArticleInteraction) {
        isBookmarked = interaction.isBookmarked
        feedbackRaw = interaction.feedback?.rawValue
        openedCount = interaction.openedCount
        lastOpenedAt = interaction.lastOpenedAt
        updatedAt = interaction.updatedAt
        payload = try? JSONEncoder().encode(interaction)
    }
}

@Model
final class PersistentArticleSnapshotRecord {
    @Attribute(.unique) var id: String
    var payload: Data
    var updatedAt: Date

    init(id: String, payload: Data, updatedAt: Date) {
        self.id = id
        self.payload = payload
        self.updatedAt = updatedAt
    }
}

@Model
final class PersistentRankedArticleRecord {
    @Attribute(.unique) var id: String
    var sourceID: String
    var title: String
    var link: String
    var categoryRaw: String
    var score: Double
    var publishedAt: Date?
    var refreshedAt: Date
    var payload: Data

    init(article: ScoredArticle, payload: Data, refreshedAt: Date) {
        id = article.id
        sourceID = article.article.sourceID
        title = article.article.title
        link = article.article.link
        categoryRaw = article.category.rawValue
        score = article.score
        publishedAt = article.article.publishedAt
        self.refreshedAt = refreshedAt
        self.payload = payload
    }

    func apply(article: ScoredArticle, payload: Data, refreshedAt: Date) {
        sourceID = article.article.sourceID
        title = article.article.title
        link = article.article.link
        categoryRaw = article.category.rawValue
        score = article.score
        publishedAt = article.article.publishedAt
        self.refreshedAt = refreshedAt
        self.payload = payload
    }
}
