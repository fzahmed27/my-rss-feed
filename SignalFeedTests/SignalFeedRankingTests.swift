import XCTest
@testable import SignalFeed

final class SignalFeedRankingTests: XCTestCase {
    private let now = Date(timeIntervalSince1970: 1_800_000_000)

    func testRankingChangesByIntentMode() {
        let articles = [
            article(id: "controls", title: "PLC control systems gain embedded AI edge inference", summary: "Industrial automation factory controls and PID tuning."),
            article(id: "space", title: "NASA satellite science mission finds new materials", summary: "Space and science update from a primary source.")
        ]

        let buildRanked = RankingEngine.rank(articles: articles, settings: .testDefaults, intent: .buildControlsAI, now: now)
        let spaceRanked = RankingEngine.rank(articles: articles, settings: .testDefaults, intent: .spaceAndScience, now: now)

        XCTAssertEqual(buildRanked.first?.article.id, "controls")
        let buildSpaceScore = buildRanked.first { $0.article.id == "space" }?.score ?? 0
        let scienceSpaceScore = spaceRanked.first { $0.article.id == "space" }?.score ?? 0

        XCTAssertGreaterThan(scienceSpaceScore, buildSpaceScore)
    }

    func testIndustrialAIPreferencesBeatGenericConsumerAI() {
        let articles = [
            article(id: "industrial", title: "Predictive maintenance for PLC sensor systems", summary: "Edge inference for manufacturing reliability."),
            article(id: "generic", title: "Generic consumer AI chatbot drama announces model launch", summary: "A consumer AI announcement with little technical depth.")
        ]

        let ranked = RankingEngine.rank(articles: articles, settings: .testDefaults, intent: .buildControlsAI, now: now)

        XCTAssertEqual(ranked.first?.article.id, "industrial")
        XCTAssertTrue((ranked.first?.score ?? 0) > (ranked.last?.score ?? 0))
    }

    func testDuplicateTopicPenalty() {
        let articles = [
            article(id: "first", title: "Industrial automation PLC edge inference launch", summary: "Controls AI for factories."),
            article(id: "second", title: "Industrial automation PLC edge inference announced", summary: "Controls AI for factories from another source.", link: "https://example.com/two")
        ]

        let ranked = RankingEngine.rank(articles: articles, settings: .testDefaults, intent: .buildControlsAI, now: now)

        XCTAssertTrue(ranked.contains { $0.penalties.contains { $0.title == "Duplicate topic" } })
    }

    func testEventClusterSelectsHighestReputationSourceAndExposesAlternates() throws {
        let lowerTrust = article(
            id: "lower-trust",
            title: "Industrial automation PLC edge inference launch",
            summary: "A factory controls platform adds edge inference.",
            reputation: 0.95,
            sourceName: "Industry Blog"
        )
        let primary = article(
            id: "primary",
            title: "Industrial automation PLC edge inference released",
            summary: "A factory controls platform adds edge inference.",
            reputation: 1.35,
            sourceName: "Primary Vendor"
        )

        let ranked = RankingEngine.rank(
            articles: [lowerTrust, primary],
            settings: .testDefaults,
            intent: .buildControlsAI,
            now: now
        )

        let representative = try XCTUnwrap(ranked.first)
        XCTAssertEqual(ranked.count, 1)
        XCTAssertEqual(representative.article.id, "primary")
        XCTAssertEqual(representative.eventCluster?.sourceCount, 2)
        XCTAssertEqual(representative.eventCluster?.alternateArticles.map(\.id), ["lower-trust"])
    }

    func testEventClusteringIsDeterministicAcrossInputOrder() {
        let articles = [
            article(id: "one", title: "PLC control platform adds edge inference", summary: "Industrial factory update.", reputation: 1.0),
            article(id: "two", title: "PLC control platform gains edge inference", summary: "Industrial factory update.", reputation: 1.2),
            article(id: "unrelated", title: "NASA publishes satellite materials data", summary: "Space science update.", reputation: 1.1)
        ]

        let first = RankingEngine.rank(articles: articles, settings: .testDefaults, now: now)
        let reversed = RankingEngine.rank(articles: Array(articles.reversed()), settings: .testDefaults, now: now)

        XCTAssertEqual(first.map(\.id), reversed.map(\.id))
        XCTAssertEqual(first.compactMap(\.eventCluster?.id), reversed.compactMap(\.eventCluster?.id))
    }

    func testEditableFounderProfileChangesRanking() throws {
        let customProfile = FounderContextProfile(
            company: "Acme Motion",
            customers: ["precision foundries"],
            productAreas: ["quantum actuation"],
            competitors: ["Vector Dynamics"],
            priorities: ["ultrafast calibration"],
            topics: ["flux sensing"]
        )
        let candidate = article(
            id: "custom-profile",
            title: "Quantum actuation and flux sensing reach precision foundries",
            summary: "Acme Motion evaluates ultrafast calibration."
        )

        let baseline = try XCTUnwrap(RankingEngine.rank(
            articles: [candidate],
            settings: .testDefaults,
            preferenceProfile: .industrialAI,
            now: now
        ).first)
        let personalized = try XCTUnwrap(RankingEngine.rank(
            articles: [candidate],
            settings: .testDefaults,
            preferenceProfile: customProfile.rankingPreferenceProfile,
            now: now
        ).first)

        XCTAssertGreaterThan(personalized.score, baseline.score)
        XCTAssertTrue(personalized.matchedKeywords.contains("quantum actuation"))
        XCTAssertTrue(personalized.matchedKeywords.contains("flux sensing"))
    }

    func testMutedTopicsAreFiltered() {
        let muteExpiry = now.addingTimeInterval(30 * 86_400)
        let learning = RankingLearningProfile(
            sourceAdjustments: [:],
            keywordAdjustments: [:],
            topicAdjustments: [:],
            mutedTopics: ["plc": muteExpiry]
        )

        let ranked = RankingEngine.rank(
            articles: [article(id: "muted", title: "PLC control systems update", summary: "Industrial automation.")],
            settings: .testDefaults,
            intent: .buildControlsAI,
            learningProfile: learning,
            now: now
        )

        XCTAssertTrue(ranked.isEmpty)
    }

    func testFeedbackReasonsPersistThroughCodableInteraction() throws {
        let interaction = ArticleInteraction(
            isBookmarked: true,
            feedback: .liked,
            feedbackReasons: [.relevantToCurrentProject, .deepTechnicalInsight],
            isRead: true,
            isDismissed: false,
            openedCount: 2,
            updatedAt: now
        )

        let data = try JSONEncoder().encode(interaction)
        let decoded = try JSONDecoder().decode(ArticleInteraction.self, from: data)

        XCTAssertEqual(decoded.feedbackReasons, [.relevantToCurrentProject, .deepTechnicalInsight])
        XCTAssertTrue(decoded.isRead)
    }

    func testReadingBudgetGroupsArticles() {
        let ranked = (0..<20).map { index in
            scored(id: "article-\(index)", score: Double(100 - index))
        }
        let budget = ReadingBudget(
            articles: ranked,
            interactions: [
                "article-0": ArticleInteraction(isBookmarked: false, feedback: nil, isRead: true, updatedAt: now),
                "article-5": ArticleInteraction(isBookmarked: false, feedback: nil, isRead: true, updatedAt: now)
            ]
        )

        XCTAssertEqual(budget.mustRead.count, 5)
        XCTAssertEqual(budget.worthSkimming.count, 10)
        XCTAssertEqual(budget.more.count, 5)
        XCTAssertEqual(budget.mustReadReadCount, 1)
        XCTAssertEqual(budget.worthSkimmingReadCount, 1)
    }

    func testFounderBriefingHasExactlyFiveSlotsAndEmptyStates() {
        let items = FounderBriefingGenerator.generate(from: [
            scored(id: "ai", title: "AI agent benchmark improves edge inference", summary: "Model capability update."),
            scored(id: "automation", title: "PLC sensor predictive maintenance for factories", summary: "Industrial automation."),
            scored(id: "tools", title: "Open source SDK for AI developer tools", summary: "Developer tooling."),
            scored(id: "startup", title: "Industrial startup signs enterprise customer", summary: "Commercialization and go-to-market.")
        ])

        XCTAssertEqual(items.count, 5)
        XCTAssertTrue(items.contains { $0.slot == .scienceOrSpace && $0.article == nil })
    }

    func testScoreExplanationGeneration() throws {
        let ranked = RankingEngine.rank(
            articles: [article(id: "explain", title: "Embedded AI sensor edge inference for industrial automation", summary: "Predictive maintenance for control systems.")],
            settings: .testDefaults,
            intent: .buildControlsAI,
            now: now
        )

        let first = try XCTUnwrap(ranked.first)
        XCTAssertFalse(first.scoreComponents.isEmpty)
        XCTAssertTrue(first.decisionSummary.whyThisMatters.hasPrefix("Why this matters:"))
        XCTAssertGreaterThan(first.decisionSummary.estimatedReadingMinutes, 0)
        XCTAssertFalse(first.decisionSummary.shouldCare.isEmpty)
        XCTAssertFalse(first.decisionSummary.evidence.isEmpty)
        XCTAssertFalse(first.decisionSummary.primaryUncertainty.isEmpty)
    }

    func testPresentationSettingsDefaultToTwentyMinuteBudget() throws {
        let settings = FeedPresentationSettings.defaults
        XCTAssertEqual(settings.selectedIntent, .buildControlsAI)
        XCTAssertEqual(settings.readingBudgetMinutes, 20)

        let data = #"{"selectedIntent":"marketIntelligence"}"#.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(FeedPresentationSettings.self, from: data)
        XCTAssertEqual(decoded.readingBudgetMinutes, 20)
    }

    private func article(
        id: String,
        title: String,
        summary: String,
        link: String? = nil,
        reputation: Double = 1.15,
        sourceName: String = "Primary Source"
    ) -> FeedArticle {
        FeedArticle(
            id: id,
            sourceID: "source-\(id)",
            sourceName: sourceName,
            sourceReputation: reputation,
            sourceKind: .feed,
            title: title,
            link: link ?? "https://example.com/\(id)",
            summary: summary,
            content: summary,
            extractedContent: String(repeating: "\(summary) ", count: 30),
            publishedAt: now.addingTimeInterval(-3_600)
        )
    }

    private func scored(id: String, score: Double = 80, title: String? = nil, summary: String = "Industrial automation signal.") -> ScoredArticle {
        let article = article(id: id, title: title ?? id, summary: summary)
        return RankingEngine.rank(
            articles: [article],
            settings: .testDefaults,
            intent: .buildControlsAI,
            now: now
        ).first ?? ScoredArticle.placeholder
    }
}

private extension FeedSettings {
    static let testDefaults = FeedSettings(
        minimumScore: 0,
        daysBack: 90,
        maxArticles: 100,
        isFullTextExtractionEnabled: false,
        fullTextArticleLimit: 0
    )
}
