import XCTest
@testable import SignalFeed

@MainActor
final class FounderModePreviewFixtureTests: XCTestCase {
    func testFreshFeedFixtureStartsUnread() throws {
        let store = FounderModePreviewFixtures.freshFeed
        let result = try XCTUnwrap(store.result)

        XCTAssertEqual(result.articles.count, 20)
        XCTAssertTrue(result.articles.allSatisfy { !store.isRead($0) })
    }

    func testCaughtUpFixtureCompletesReadingBudget() throws {
        let store = FounderModePreviewFixtures.caughtUpFeed
        let result = try XCTUnwrap(store.result)
        let budget = ReadingBudget(articles: result.articles, interactions: store.interactions)

        XCTAssertEqual(budget.mustReadReadCount, 5)
        XCTAssertEqual(budget.worthSkimmingReadCount, 10)
        XCTAssertTrue(budget.isCaughtUp)
    }

    func testNoHighQualityFixtureRepresentsFilteredFetch() throws {
        let result = try XCTUnwrap(FounderModePreviewFixtures.noHighQualityFeed.result)

        XCTAssertTrue(result.articles.isEmpty)
        XCTAssertEqual(result.fetchedCount, 24)
    }

    func testMutedTopicFixtureIsActiveAndFiltersMatchingArticles() throws {
        let store = FounderModePreviewFixtures.mutedTopicFeed
        let result = try XCTUnwrap(store.result)

        XCTAssertEqual(store.mutedTopics["plc"]?.topic, "PLC")
        XCTAssertTrue(store.mutedTopics["plc"]?.isActive(at: FounderModePreviewFixtures.referenceDate) == true)
        XCTAssertFalse(result.articles.contains { $0.article.searchableText.contains("plc") })
    }

    func testLongTitleFixtureExercisesCardWrapping() {
        XCTAssertGreaterThan(FounderModePreviewFixtures.longTitleArticle.article.title.count, 100)
    }
}
