import Foundation

struct FeedSource: Identifiable, Hashable, Codable, Sendable {
    var id: String
    var name: String
    var url: String
    var reputation: Double
    var kind: SourceKind
    var isEnabled: Bool
    var isMuted: Bool

    init(
        id: String,
        name: String,
        url: String,
        reputation: Double,
        kind: SourceKind = .feed,
        isEnabled: Bool = true,
        isMuted: Bool = false
    ) {
        self.id = id
        self.name = name
        self.url = url
        self.reputation = reputation
        self.kind = kind
        self.isEnabled = isEnabled
        self.isMuted = isMuted
    }

    var isActive: Bool {
        isEnabled && !isMuted
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case name
        case url
        case reputation
        case kind
        case isEnabled
        case isMuted
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        name = try container.decode(String.self, forKey: .name)
        url = try container.decode(String.self, forKey: .url)
        reputation = try container.decode(Double.self, forKey: .reputation)
        kind = try container.decodeIfPresent(SourceKind.self, forKey: .kind) ?? .feed
        isEnabled = try container.decodeIfPresent(Bool.self, forKey: .isEnabled) ?? true
        isMuted = try container.decodeIfPresent(Bool.self, forKey: .isMuted) ?? false
    }
}

enum SourceKind: String, CaseIterable, Codable, Identifiable, Sendable {
    case feed
    case person

    var id: String { rawValue }

    var title: String {
        switch self {
        case .feed: "Publication"
        case .person: "Person"
        }
    }

    var systemImage: String {
        switch self {
        case .feed: "newspaper"
        case .person: "person.crop.circle"
        }
    }
}

enum ArticleFeedback: String, CaseIterable, Codable, Identifiable, Sendable {
    case liked
    case disliked

    var id: String { rawValue }

    var title: String {
        switch self {
        case .liked: "Useful"
        case .disliked: "Not useful"
        }
    }

    var systemImage: String {
        switch self {
        case .liked: "hand.thumbsup.fill"
        case .disliked: "hand.thumbsdown.fill"
        }
    }
}

enum FeedbackReason: String, CaseIterable, Codable, Identifiable, Sendable {
    case relevantToCurrentProject
    case strongBusinessSignal
    case deepTechnicalInsight
    case trackThisTopic
    case trackThisCompanyOrPerson
    case tooGeneric
    case notRelevantToMyCompany
    case tooAcademic
    case alreadyKnewThis
    case lowQualitySource
    case interestingButNotNow

    var id: String { rawValue }

    var title: String {
        switch self {
        case .relevantToCurrentProject: "Relevant to current project"
        case .strongBusinessSignal: "Strong business signal"
        case .deepTechnicalInsight: "Deep technical insight"
        case .trackThisTopic: "Track this topic"
        case .trackThisCompanyOrPerson: "Track this company or person"
        case .tooGeneric: "Too generic"
        case .notRelevantToMyCompany: "Not relevant to my company"
        case .tooAcademic: "Too academic"
        case .alreadyKnewThis: "Already knew this"
        case .lowQualitySource: "Low-quality source"
        case .interestingButNotNow: "Interesting, but not now"
        }
    }

    var adjustment: Double {
        switch self {
        case .relevantToCurrentProject: 0.55
        case .strongBusinessSignal: 0.45
        case .deepTechnicalInsight: 0.45
        case .trackThisTopic: 0.35
        case .trackThisCompanyOrPerson: 0.3
        case .tooGeneric: -0.45
        case .notRelevantToMyCompany: -0.55
        case .tooAcademic: -0.35
        case .alreadyKnewThis: -0.25
        case .lowQualitySource: -0.45
        case .interestingButNotNow: -0.2
        }
    }
}

struct ArticleInteraction: Codable, Hashable, Sendable {
    var isBookmarked: Bool
    var feedback: ArticleFeedback?
    var feedbackReasons: [FeedbackReason]
    var isRead: Bool
    var isDismissed: Bool
    var openedCount: Int
    var lastOpenedAt: Date?
    var updatedAt: Date

    static let empty = ArticleInteraction(
        isBookmarked: false,
        feedback: nil,
        feedbackReasons: [],
        isRead: false,
        isDismissed: false,
        openedCount: 0,
        lastOpenedAt: nil,
        updatedAt: Date(timeIntervalSince1970: 0)
    )

    init(
        isBookmarked: Bool,
        feedback: ArticleFeedback?,
        feedbackReasons: [FeedbackReason] = [],
        isRead: Bool = false,
        isDismissed: Bool = false,
        openedCount: Int = 0,
        lastOpenedAt: Date? = nil,
        updatedAt: Date
    ) {
        self.isBookmarked = isBookmarked
        self.feedback = feedback
        self.feedbackReasons = feedbackReasons
        self.isRead = isRead
        self.isDismissed = isDismissed
        self.openedCount = openedCount
        self.lastOpenedAt = lastOpenedAt
        self.updatedAt = updatedAt
    }

    private enum CodingKeys: String, CodingKey {
        case isBookmarked
        case feedback
        case feedbackReasons
        case isRead
        case isDismissed
        case openedCount
        case lastOpenedAt
        case updatedAt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        isBookmarked = try container.decodeIfPresent(Bool.self, forKey: .isBookmarked) ?? false
        feedback = try container.decodeIfPresent(ArticleFeedback.self, forKey: .feedback)
        feedbackReasons = try container.decodeIfPresent([FeedbackReason].self, forKey: .feedbackReasons) ?? []
        isRead = try container.decodeIfPresent(Bool.self, forKey: .isRead) ?? false
        isDismissed = try container.decodeIfPresent(Bool.self, forKey: .isDismissed) ?? false
        openedCount = try container.decodeIfPresent(Int.self, forKey: .openedCount) ?? 0
        lastOpenedAt = try container.decodeIfPresent(Date.self, forKey: .lastOpenedAt)
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date(timeIntervalSince1970: 0)
    }
}

enum RankingReasonKind: String, CaseIterable, Codable, Identifiable, Sendable {
    case sourceTrust
    case keyword
    case fullText
    case freshness
    case opportunity
    case duplicate
    case personal
    case baseline

    var id: String { rawValue }

    var title: String {
        switch self {
        case .sourceTrust: "Source trust"
        case .keyword: "Keyword hit"
        case .fullText: "Full text"
        case .freshness: "Freshness"
        case .opportunity: "Opportunity flag"
        case .duplicate: "Duplicate penalty"
        case .personal: "Personal learning"
        case .baseline: "Baseline"
        }
    }

    var systemImage: String {
        switch self {
        case .sourceTrust: "checkmark.seal"
        case .keyword: "tag"
        case .fullText: "doc.text.magnifyingglass"
        case .freshness: "clock"
        case .opportunity: "sparkles"
        case .duplicate: "doc.on.doc"
        case .personal: "person.crop.circle.badge.checkmark"
        case .baseline: "slider.horizontal.3"
        }
    }
}

enum IntentMode: String, CaseIterable, Codable, Identifiable, Sendable {
    case buildControlsAI
    case learn
    case marketIntelligence
    case spaceAndScience
    case quickCatchUp

    var id: String { rawValue }

    var title: String {
        switch self {
        case .buildControlsAI: "Build Controls AI"
        case .learn: "Learn"
        case .marketIntelligence: "Market Intelligence"
        case .spaceAndScience: "Space & Science"
        case .quickCatchUp: "Quick Catch-Up"
        }
    }
}

enum ScoreComponentKind: String, CaseIterable, Codable, Identifiable, Sendable {
    case personalRelevance
    case actionability
    case strategicImportance
    case sourceQuality
    case novelty
    case recency

    var id: String { rawValue }

    var title: String {
        switch self {
        case .personalRelevance: "Personal relevance"
        case .actionability: "Actionability"
        case .strategicImportance: "Strategic importance"
        case .sourceQuality: "Source quality"
        case .novelty: "Novelty"
        case .recency: "Recency"
        }
    }
}

struct ScoreComponent: Identifiable, Hashable, Codable, Sendable {
    let kind: ScoreComponentKind
    let rawValue: Double
    let normalizedValue: Double
    let weight: Double
    let contribution: Double
    let explanation: String

    var id: String { kind.rawValue }
}

struct ScorePenalty: Identifiable, Hashable, Codable, Sendable {
    let title: String
    let value: Double
    let explanation: String

    var id: String { title }
}

enum SignalConfidence: String, CaseIterable, Codable, Identifiable, Sendable {
    case high
    case medium
    case low

    var id: String { rawValue }

    var title: String {
        switch self {
        case .high: "High confidence"
        case .medium: "Medium confidence"
        case .low: "Low confidence"
        }
    }
}

struct ArticleDecisionSummary: Hashable, Codable, Sendable {
    let whyThisMatters: String
    let whatChanged: String
    let whyItMattersToYou: String
    let shouldCare: String
    let suggestedAction: String
    let confidence: SignalConfidence
    let primaryUncertainty: String
    let estimatedReadingMinutes: Int
    let evidence: String
}

struct MutedTopic: Identifiable, Hashable, Codable, Sendable {
    let topic: String
    var expiresAt: Date
    var createdAt: Date

    var id: String { topic.lowercased() }

    func isActive(at date: Date = Date()) -> Bool {
        expiresAt > date
    }
}

struct FounderPreferenceProfile: Hashable, Codable, Sendable {
    var topicWeights: [String: Double]
    var penaltyWeights: [String: Double]
    var intentWeights: [IntentMode: [String: Double]]
    var componentWeights: [ScoreComponentKind: Double]

    static let industrialAI = FounderPreferenceProfile(
        topicWeights: [
            "industrial automation": 1.0,
            "plc": 1.0,
            "plcs": 1.0,
            "pid tuning": 1.0,
            "control systems": 0.95,
            "embedded ai": 0.95,
            "edge inference": 0.95,
            "robotics for manufacturing": 0.9,
            "manufacturing robotics": 0.9,
            "sensors": 0.8,
            "sensor": 0.75,
            "signal processing": 0.85,
            "predictive maintenance": 0.9,
            "ai developer tools": 0.8,
            "developer tools": 0.75,
            "industrial startup": 0.85,
            "commercialization": 0.75,
            "go-to-market": 0.75,
            "enterprise adoption": 0.75,
            "factory": 0.65,
            "robotics": 0.55,
            "edge ai": 0.8
        ],
        penaltyWeights: [
            "generic consumer ai": 0.75,
            "ai drama": 0.85,
            "model launch": 0.45,
            "funding": 0.35,
            "generic robotics paper": 0.4,
            "duplicate": 0.65,
            "celebrity": 0.5,
            "chatbot": 0.3
        ],
        intentWeights: [
            .buildControlsAI: [
                "industrial automation": 1.0,
                "plc": 1.0,
                "pid tuning": 1.0,
                "control systems": 1.0,
                "embedded ai": 0.9,
                "edge inference": 0.9,
                "predictive maintenance": 0.8
            ],
            .learn: [
                "deep technical insight": 1.0,
                "tutorial": 0.75,
                "research": 0.6,
                "architecture": 0.6,
                "control systems": 0.6
            ],
            .marketIntelligence: [
                "commercialization": 1.0,
                "go-to-market": 1.0,
                "customer": 0.8,
                "procurement": 0.8,
                "earnings": 0.8,
                "guidance": 0.75
            ],
            .spaceAndScience: [
                "space": 1.0,
                "science": 0.9,
                "nasa": 0.9,
                "satellite": 0.8,
                "physics": 0.6,
                "materials": 0.6
            ],
            .quickCatchUp: [
                "launch": 0.55,
                "released": 0.55,
                "summary": 0.4,
                "what changed": 0.4,
                "market": 0.35
            ]
        ],
        componentWeights: [
            .personalRelevance: 0.30,
            .actionability: 0.25,
            .strategicImportance: 0.20,
            .sourceQuality: 0.10,
            .novelty: 0.10,
            .recency: 0.05
        ]
    )
}

struct FounderContextProfile: Hashable, Codable, Sendable {
    var company: String
    var customers: [String]
    var productAreas: [String]
    var competitors: [String]
    var priorities: [String]
    var topics: [String]

    static let industrialAIFounder = FounderContextProfile(
        company: "Industrial AI company",
        customers: ["manufacturers", "factory operators", "automation teams"],
        productAreas: ["industrial automation", "control systems", "embedded AI", "edge inference"],
        competitors: [],
        priorities: ["commercialization", "go-to-market", "predictive maintenance"],
        topics: ["PLCs", "PID tuning", "manufacturing robotics", "sensors", "signal processing"]
    )

    static let rankingFieldWeights: [KeyPath<FounderContextProfile, [String]>: Double] = [
        \.customers: 0.70,
        \.productAreas: 1.00,
        \.competitors: 0.55,
        \.priorities: 0.90,
        \.topics: 0.85
    ]

    var rankingPreferenceProfile: FounderPreferenceProfile {
        var preference = FounderPreferenceProfile.industrialAI
        let companyTerm = company.normalizedProfileTerm
        if !companyTerm.isEmpty {
            preference.topicWeights[companyTerm] = max(preference.topicWeights[companyTerm, default: 0], 0.85)
        }

        for (keyPath, weight) in Self.rankingFieldWeights {
            for value in self[keyPath: keyPath] {
                let term = value.normalizedProfileTerm
                guard !term.isEmpty else { continue }
                preference.topicWeights[term] = max(preference.topicWeights[term, default: 0], weight)
            }
        }
        return preference
    }

    var exportText: String {
        [
            "Signal Feed founder profile",
            "Company: \(company)",
            "Customers: \(customers.joined(separator: ", "))",
            "Product areas: \(productAreas.joined(separator: ", "))",
            "Competitors: \(competitors.joined(separator: ", "))",
            "Priorities: \(priorities.joined(separator: ", "))",
            "Topics: \(topics.joined(separator: ", "))"
        ].joined(separator: "\n")
    }
}

private extension String {
    var normalizedProfileTerm: String {
        trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }
}

struct FeedPresentationSettings: Codable, Equatable, Sendable {
    var selectedIntent: IntentMode
    var readingBudgetMinutes: Int

    static let defaults = FeedPresentationSettings(selectedIntent: .buildControlsAI, readingBudgetMinutes: 20)

    private enum CodingKeys: String, CodingKey {
        case selectedIntent
        case readingBudgetMinutes
    }

    init(selectedIntent: IntentMode, readingBudgetMinutes: Int) {
        self.selectedIntent = selectedIntent
        self.readingBudgetMinutes = readingBudgetMinutes
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        selectedIntent = try container.decodeIfPresent(IntentMode.self, forKey: .selectedIntent) ?? Self.defaults.selectedIntent
        readingBudgetMinutes = try container.decodeIfPresent(Int.self, forKey: .readingBudgetMinutes) ?? Self.defaults.readingBudgetMinutes
    }
}

struct SourceHealthSummary: Codable, Hashable, Sendable {
    var lastFetchAt: Date?
    var lastSuccessfulFetchAt: Date?
    var failureStreak: Int
    var totalFetches: Int
    var averageQuality: Double
    var lastStatus: SourceStatus?
    var lastMessage: String
    var lastItemCount: Int
    var lastDuration: TimeInterval

    static let empty = SourceHealthSummary(
        lastFetchAt: nil,
        lastSuccessfulFetchAt: nil,
        failureStreak: 0,
        totalFetches: 0,
        averageQuality: 0,
        lastStatus: nil,
        lastMessage: "",
        lastItemCount: 0,
        lastDuration: 0
    )
}

struct RankingLearningProfile: Hashable, Sendable {
    var sourceAdjustments: [String: Double]
    var keywordAdjustments: [String: Double]
    var topicAdjustments: [String: Double]
    var mutedTopics: [String: Date]

    static let empty = RankingLearningProfile(sourceAdjustments: [:], keywordAdjustments: [:], topicAdjustments: [:], mutedTopics: [:])
}

struct RankingReason: Identifiable, Hashable, Codable, Sendable {
    let kind: RankingReasonKind
    let title: String
    let detail: String
    let impact: Double

    var id: String {
        "\(kind.rawValue)-\(title)-\(detail)-\(impact)"
    }
}

enum ArticleCategory: String, CaseIterable, Codable, Identifiable, Sendable {
    case people
    case tactile
    case vision
    case robotics
    case ai
    case hardware
    case markets
    case startups
    case other

    var id: String { rawValue }

    var title: String {
        switch self {
        case .people: "People"
        case .tactile: "Tactile & Haptics"
        case .vision: "Computer Vision"
        case .robotics: "Robotics"
        case .ai: "AI & ML"
        case .hardware: "Hardware"
        case .markets: "Markets"
        case .startups: "Startups"
        case .other: "Other"
        }
    }

    var systemImage: String {
        switch self {
        case .people: "person.2"
        case .tactile: "hand.tap"
        case .vision: "camera.macro"
        case .robotics: "gearshape.2"
        case .ai: "sparkles"
        case .hardware: "cpu"
        case .markets: "chart.line.uptrend.xyaxis"
        case .startups: "building.2"
        case .other: "tray"
        }
    }
}

enum OpportunityLabel: String, CaseIterable, Codable, Identifiable, Sendable {
    case marketMoving
    case businessOpportunity
    case aiLaunch

    var id: String { rawValue }

    var title: String {
        switch self {
        case .marketMoving: "Market-moving"
        case .businessOpportunity: "Business opportunity"
        case .aiLaunch: "AI launch"
        }
    }

    var systemImage: String {
        switch self {
        case .marketMoving: "chart.bar.xaxis"
        case .businessOpportunity: "briefcase"
        case .aiLaunch: "wand.and.stars"
        }
    }
}

struct FeedArticle: Identifiable, Hashable, Codable, Sendable {
    let id: String
    let sourceID: String
    let sourceName: String
    let sourceReputation: Double
    let sourceKind: SourceKind
    let title: String
    let link: String
    let summary: String
    let content: String
    let extractedContent: String
    let publishedAt: Date?

    var searchableText: String {
        [title, sourceName, summary, content, extractedContent]
            .joined(separator: " ")
            .lowercased()
    }

    var rankingContent: String {
        extractedContent.isEmpty ? content : "\(content) \(extractedContent)"
    }

    var hasExtractedContent: Bool {
        extractedContent.count > 300
    }

    init(
        id: String,
        sourceID: String,
        sourceName: String,
        sourceReputation: Double,
        sourceKind: SourceKind,
        title: String,
        link: String,
        summary: String,
        content: String,
        extractedContent: String = "",
        publishedAt: Date?
    ) {
        self.id = id
        self.sourceID = sourceID
        self.sourceName = sourceName
        self.sourceReputation = sourceReputation
        self.sourceKind = sourceKind
        self.title = title
        self.link = link
        self.summary = summary
        self.content = content
        self.extractedContent = extractedContent
        self.publishedAt = publishedAt
    }

    func withExtractedContent(_ value: String) -> FeedArticle {
        FeedArticle(
            id: id,
            sourceID: sourceID,
            sourceName: sourceName,
            sourceReputation: sourceReputation,
            sourceKind: sourceKind,
            title: title,
            link: link,
            summary: summary,
            content: content,
            extractedContent: value,
            publishedAt: publishedAt
        )
    }

    private enum CodingKeys: String, CodingKey {
        case id
        case sourceID
        case sourceName
        case sourceReputation
        case sourceKind
        case title
        case link
        case summary
        case content
        case extractedContent
        case publishedAt
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(String.self, forKey: .id)
        sourceID = try container.decode(String.self, forKey: .sourceID)
        sourceName = try container.decode(String.self, forKey: .sourceName)
        sourceReputation = try container.decode(Double.self, forKey: .sourceReputation)
        sourceKind = try container.decodeIfPresent(SourceKind.self, forKey: .sourceKind) ?? .feed
        title = try container.decode(String.self, forKey: .title)
        link = try container.decode(String.self, forKey: .link)
        summary = try container.decode(String.self, forKey: .summary)
        content = try container.decode(String.self, forKey: .content)
        extractedContent = try container.decodeIfPresent(String.self, forKey: .extractedContent) ?? ""
        publishedAt = try container.decodeIfPresent(Date.self, forKey: .publishedAt)
    }
}

struct ScoredArticle: Identifiable, Hashable, Codable, Sendable {
    let article: FeedArticle
    let score: Double
    let scoreComponents: [ScoreComponent]
    let penalties: [ScorePenalty]
    let decisionSummary: ArticleDecisionSummary
    let category: ArticleCategory
    let reasons: [String]
    let reasonDetails: [RankingReason]
    let matchedKeywords: [String]
    let opportunityLabels: [OpportunityLabel]
    let canonicalURL: String
    var eventCluster: ArticleEventCluster? = nil

    var id: String { article.id }

    var relevanceLabel: String {
        switch score {
        case 72...: "High Relevance"
        case 45..<72: "Medium Relevance"
        default: "Low Relevance"
        }
    }
}

struct ArticleEventCluster: Identifiable, Hashable, Codable, Sendable {
    let id: String
    let alternateArticles: [FeedArticle]

    var sourceCount: Int { alternateArticles.count + 1 }
}

enum SourceStatus: String, Codable, Sendable {
    case healthy
    case empty
    case failed
    case paused

    var title: String {
        switch self {
        case .healthy: "Healthy"
        case .empty: "No new items"
        case .failed: "Failed"
        case .paused: "Paused"
        }
    }

    var systemImage: String {
        switch self {
        case .healthy: "checkmark.circle.fill"
        case .empty: "minus.circle.fill"
        case .failed: "xmark.circle.fill"
        case .paused: "pause.circle.fill"
        }
    }
}

struct SourceHealth: Identifiable, Hashable, Sendable {
    let source: FeedSource
    let status: SourceStatus
    let itemCount: Int
    let message: String
    let duration: TimeInterval

    var id: String { source.id }
}

struct DigestResult: Hashable, Sendable {
    let articles: [ScoredArticle]
    let sourceHealth: [SourceHealth]
    let generatedAt: Date
    let fetchedCount: Int

    var articlesByCategory: [ArticleCategory: [ScoredArticle]] {
        Dictionary(grouping: articles, by: \.category)
    }
}

struct FeedSettings: Codable, Equatable, Sendable {
    var minimumScore: Double
    var daysBack: Int
    var maxArticles: Int
    var isFullTextExtractionEnabled: Bool
    var fullTextArticleLimit: Int

    static let defaults = FeedSettings(
        minimumScore: 1.5,
        daysBack: 30,
        maxArticles: 80,
        isFullTextExtractionEnabled: true,
        fullTextArticleLimit: 60
    )

    init(
        minimumScore: Double,
        daysBack: Int,
        maxArticles: Int,
        isFullTextExtractionEnabled: Bool,
        fullTextArticleLimit: Int
    ) {
        self.minimumScore = minimumScore
        self.daysBack = daysBack
        self.maxArticles = maxArticles
        self.isFullTextExtractionEnabled = isFullTextExtractionEnabled
        self.fullTextArticleLimit = fullTextArticleLimit
    }

    private enum CodingKeys: String, CodingKey {
        case minimumScore
        case daysBack
        case maxArticles
        case isFullTextExtractionEnabled
        case fullTextArticleLimit
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        minimumScore = try container.decodeIfPresent(Double.self, forKey: .minimumScore) ?? Self.defaults.minimumScore
        daysBack = try container.decodeIfPresent(Int.self, forKey: .daysBack) ?? Self.defaults.daysBack
        maxArticles = try container.decodeIfPresent(Int.self, forKey: .maxArticles) ?? Self.defaults.maxArticles
        isFullTextExtractionEnabled = try container.decodeIfPresent(Bool.self, forKey: .isFullTextExtractionEnabled) ?? Self.defaults.isFullTextExtractionEnabled
        fullTextArticleLimit = try container.decodeIfPresent(Int.self, forKey: .fullTextArticleLimit) ?? Self.defaults.fullTextArticleLimit
    }
}

enum FeedLoadState: Sendable {
    case idle
    case loading
    case loaded(DigestResult)
    case failed(String)

    var result: DigestResult? {
        if case .loaded(let result) = self {
            return result
        }
        return nil
    }
}

extension ScoredArticle {
    static let placeholder = ScoredArticle(
        article: FeedArticle(
            id: "placeholder",
            sourceID: "placeholder",
            sourceName: "Source",
            sourceReputation: 1.0,
            sourceKind: .feed,
            title: "A ranked signal from a trusted feed",
            link: "https://example.com",
            summary: "A concise summary appears here with enough context to decide whether this deserves attention.",
            content: "",
            extractedContent: "A longer extracted article body appears here when the page can be fetched and simplified into readable text.",
            publishedAt: Date()
        ),
        score: 12.5,
        scoreComponents: [
            ScoreComponent(kind: .personalRelevance, rawValue: 0.8, normalizedValue: 0.8, weight: 0.3, contribution: 24, explanation: "Matches industrial AI interests."),
            ScoreComponent(kind: .actionability, rawValue: 0.5, normalizedValue: 0.5, weight: 0.25, contribution: 12.5, explanation: "Suggests a practical follow-up."),
            ScoreComponent(kind: .strategicImportance, rawValue: 0.4, normalizedValue: 0.4, weight: 0.2, contribution: 8, explanation: "Could affect product direction."),
            ScoreComponent(kind: .sourceQuality, rawValue: 0.7, normalizedValue: 0.7, weight: 0.1, contribution: 7, explanation: "Comes from a trusted source."),
            ScoreComponent(kind: .novelty, rawValue: 0.6, normalizedValue: 0.6, weight: 0.1, contribution: 6, explanation: "Appears distinct in this refresh."),
            ScoreComponent(kind: .recency, rawValue: 1.0, normalizedValue: 1.0, weight: 0.05, contribution: 5, explanation: "Published recently.")
        ],
        penalties: [],
        decisionSummary: ArticleDecisionSummary(
            whyThisMatters: "Why this matters: a practical AI signal may affect industrial automation strategy.",
            whatChanged: "A ranked signal from a trusted source surfaced in your feed.",
            whyItMattersToYou: "It matches your controls, robotics, or edge AI interests.",
            shouldCare: "Maybe: review if it maps to a current build or customer question.",
            suggestedAction: "Save it if it informs current work, otherwise dismiss it.",
            confidence: .medium,
            primaryUncertainty: "Only source metadata and available article text were analyzed.",
            estimatedReadingMinutes: 3,
            evidence: "Source: Source"
        ),
        category: .ai,
        reasons: ["Title matched AI agents", "Fresh: published today"],
        reasonDetails: [
            RankingReason(
                kind: .keyword,
                title: "Keyword hit: AI agents",
                detail: "Matched in the title and summary.",
                impact: 9.0
            ),
            RankingReason(
                kind: .freshness,
                title: "Fresh: published today",
                detail: "Recent items receive a small boost so the feed stays timely.",
                impact: 2.0
            ),
            RankingReason(
                kind: .duplicate,
                title: "Duplicate penalty: none",
                detail: "This item has a unique canonical URL in the current result set.",
                impact: 0
            )
        ],
        matchedKeywords: ["AI agents"],
        opportunityLabels: [.aiLaunch],
        canonicalURL: "https://example.com"
    )
}
