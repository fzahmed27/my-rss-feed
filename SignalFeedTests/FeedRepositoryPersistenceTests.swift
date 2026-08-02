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

    func testDeletingEverySourcePersistsAnIntentionallyEmptyList() {
        let repository = FeedRepository(inMemory: true)
        let source = FeedSource(
            id: "temporary-source",
            name: "Temporary Source",
            url: "https://example.com/feed.xml",
            reputation: 1
        )
        repository.saveSources([source], metrics: [:])
        let store = FeedStore(client: FeedClient(sources: []), repository: repository)

        store.deleteSource(source)

        XCTAssertTrue(repository.loadSources().isEmpty)
        let reloaded = FeedStore(client: FeedClient(sources: []), repository: repository)
        XCTAssertTrue(reloaded.sources.isEmpty)
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

final class RSSFeedParserTests: XCTestCase {
    func testPrefixedAtomParsesAlternateLinkCDATAAndNestedXHTML() throws {
        let xml = """
        <?xml version="1.0" encoding="utf-8"?>
        <atom:feed xmlns:atom="http://www.w3.org/2005/Atom" xmlns:xhtml="http://www.w3.org/1999/xhtml">
          <atom:entry>
            <atom:title>Nested <xhtml:b>Atom</xhtml:b> title</atom:title>
            <atom:link rel="self" href="https://example.com/api/entry/42" />
            <atom:link rel="alternate" href="https://example.com/articles/42" />
            <atom:summary><![CDATA[Factory &amp; controls summary]]></atom:summary>
            <atom:content type="xhtml">
              <xhtml:div><xhtml:p>Deep <xhtml:b>technical</xhtml:b> content</xhtml:p></xhtml:div>
            </atom:content>
            <atom:updated>2026-07-31T15:30:00Z</atom:updated>
          </atom:entry>
        </atom:feed>
        """
        let source = FeedSource(
            id: "atom-source",
            name: "Atom Source",
            url: "https://example.com/atom.xml",
            reputation: 1
        )

        let article = try XCTUnwrap(RSSFeedParser.parse(data: Data(xml.utf8), source: source).first)

        XCTAssertEqual(article.title, "Nested Atom title")
        XCTAssertEqual(article.link, "https://example.com/articles/42")
        XCTAssertEqual(article.summary, "Factory & controls summary")
        XCTAssertEqual(article.content, "Deep technical content")
        XCTAssertNotNil(article.publishedAt)
    }
}
