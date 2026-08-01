import XCTest
@testable import SignalFeed

@MainActor
final class FeedRepositoryPersistenceTests: XCTestCase {
    private let timestamp = Date(timeIntervalSince1970: 1_800_000_000)

    func testReadStateRoundTripsThroughRepositoryStorage() throws {
        let repository = FeedRepository(inMemory: true)
        let articleID = "read-state"

        repository.saveInteractions([
            articleID: ArticleInteraction(
                isBookmarked: false,
                feedback: nil,
                feedbackReasons: [],
                isRead: true,
                isDismissed: false,
                openedCount: 1,
                lastOpenedAt: timestamp,
                updatedAt: timestamp
            )
        ])

        var loaded = try XCTUnwrap(repository.loadInteractions()[articleID])
        XCTAssertTrue(loaded.isRead)
        XCTAssertEqual(loaded.lastOpenedAt, timestamp)

        loaded.isRead = false
        loaded.updatedAt = timestamp.addingTimeInterval(60)
        repository.saveInteractions([articleID: loaded])

        let reloaded = try XCTUnwrap(repository.loadInteractions()[articleID])
        XCTAssertFalse(reloaded.isRead)
        XCTAssertEqual(reloaded.updatedAt, timestamp.addingTimeInterval(60))
    }

    func testMutedTopicsExpireAndReloadCorrectly() {
        let repository = FeedRepository(inMemory: true)
        let active = MutedTopic(
            topic: "PLC",
            expiresAt: Date().addingTimeInterval(86_400),
            createdAt: timestamp
        )
        let expired = MutedTopic(
            topic: "Consumer AI",
            expiresAt: Date().addingTimeInterval(-86_400),
            createdAt: timestamp
        )

        repository.saveMutedTopics([active.id: active, expired.id: expired])

        XCTAssertEqual(repository.loadMutedTopics(), [active.id: active])
    }

    func testPresentationSettingsRoundTripThroughRepositoryStorage() {
        let repository = FeedRepository(inMemory: true)
        let settings = FeedPresentationSettings(
            selectedIntent: .marketIntelligence,
            readingBudgetMinutes: 35
        )

        repository.savePresentationSettings(settings)

        XCTAssertEqual(repository.loadPresentationSettings(), settings)
    }

    func testFeedbackExportContainsInspectableInteractionFields() throws {
        let repository = FeedRepository(inMemory: true)
        let store = FeedStore(
            client: FeedClient(sources: []),
            repository: repository,
            sources: []
        )
        let article = makeArticle()

        store.setFeedback(
            .liked,
            reasons: [.relevantToCurrentProject, .deepTechnicalInsight],
            for: article
        )

        let interaction = store.interaction(for: article)
        let exportURL = try store.makeFeedbackExportURL()
        addTeardownBlock {
            try? FileManager.default.removeItem(at: exportURL)
        }
        let export = try String(contentsOf: exportURL, encoding: .utf8)

        XCTAssertTrue(export.contains("- PLC edge inference cuts factory downtime"))
        XCTAssertTrue(export.contains("feedback: liked"))
        XCTAssertTrue(export.contains("reasons: Relevant to current project, Deep technical insight"))
        XCTAssertTrue(export.contains("updated: \(ISO8601DateFormatter().string(from: interaction.updatedAt))"))

        let persisted = try XCTUnwrap(repository.loadInteractions()[article.id])
        XCTAssertEqual(persisted.feedback, .liked)
        XCTAssertEqual(persisted.feedbackReasons, [.relevantToCurrentProject, .deepTechnicalInsight])
    }

    func testFounderProfilePersistsExportsAndResets() throws {
        let repository = FeedRepository(inMemory: true)
        let store = FeedStore(
            client: FeedClient(sources: []),
            repository: repository,
            sources: []
        )
        let profile = FounderContextProfile(
            company: "Acme Controls",
            customers: ["steel mills"],
            productAreas: ["motor control"],
            competitors: ["Vector Automation"],
            priorities: ["pilot conversion"],
            topics: ["servo tuning"]
        )

        store.updateFounderProfile(profile)

        XCTAssertEqual(repository.loadFounderProfile(), profile)
        let reloaded = FeedStore(client: FeedClient(sources: []), repository: repository, sources: [])
        XCTAssertEqual(reloaded.founderProfile, profile)

        let exportURL = try reloaded.makeFounderProfileExportURL()
        addTeardownBlock { try? FileManager.default.removeItem(at: exportURL) }
        let export = try String(contentsOf: exportURL, encoding: .utf8)
        XCTAssertTrue(export.contains("Company: Acme Controls"))
        XCTAssertTrue(export.contains("Topics: servo tuning"))

        reloaded.resetFounderProfile()
        XCTAssertEqual(reloaded.founderProfile, .industrialAIFounder)
        XCTAssertEqual(repository.loadFounderProfile(), .industrialAIFounder)
    }

    private func makeArticle() -> ScoredArticle {
        let article = FeedArticle(
            id: "feedback-export",
            sourceID: "primary-source",
            sourceName: "Primary Source",
            sourceReputation: 1.2,
            sourceKind: .feed,
            title: "PLC edge inference cuts factory downtime",
            link: "https://example.com/feedback-export",
            summary: "Industrial automation and predictive maintenance update.",
            content: "Industrial automation and predictive maintenance update.",
            extractedContent: String(repeating: "Industrial automation and predictive maintenance. ", count: 30),
            publishedAt: timestamp
        )

        return RankingEngine.rank(
            articles: [article],
            settings: FeedSettings(
                minimumScore: 0,
                daysBack: 90,
                maxArticles: 10,
                isFullTextExtractionEnabled: false,
                fullTextArticleLimit: 0
            ),
            intent: .buildControlsAI,
            now: timestamp
        ).first ?? .placeholder
    }
}
