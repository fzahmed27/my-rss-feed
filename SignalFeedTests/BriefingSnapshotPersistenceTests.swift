import XCTest
@testable import SignalFeed

@MainActor
final class BriefingSnapshotPersistenceTests: XCTestCase {
    private let savedAt = Date(timeIntervalSince1970: 1_800_100_000)

    func testSnapshotRoundTripsWithFiveSlotsAndMetadata() throws {
        let repository = FeedRepository(inMemory: true)
        let result = try XCTUnwrap(FounderModePreviewFixtures.freshFeed.result)
        let snapshot = FounderBriefingSnapshot(
            generatedAt: result.generatedAt,
            savedAt: savedAt,
            intent: .buildControlsAI,
            readingBudgetMinutes: 20,
            fetchedCount: result.fetchedCount,
            items: FounderBriefingGenerator.generate(from: result.articles)
        )

        repository.saveBriefingSnapshots([snapshot])

        let loaded = try XCTUnwrap(repository.loadBriefingSnapshots().first)
        XCTAssertEqual(loaded, snapshot)
        XCTAssertEqual(loaded.items.count, 5)
        XCTAssertEqual(loaded.items.map(\.slot), FounderBriefingSlot.allCases)
    }

    func testStoreSavesDistinctBriefingOnceAndReloadsIt() throws {
        let repository = FeedRepository(inMemory: true)
        let result = try XCTUnwrap(FounderModePreviewFixtures.freshFeed.result)
        let store = makeStore(repository: repository, result: result)

        let first = try XCTUnwrap(store.saveCurrentBriefingSnapshot(savedAt: savedAt))
        let duplicate = try XCTUnwrap(store.saveCurrentBriefingSnapshot(savedAt: savedAt.addingTimeInterval(60)))

        XCTAssertEqual(first.id, duplicate.id)
        XCTAssertEqual(store.briefingSnapshots.count, 1)

        let reloadedStore = FeedStore(
            client: FeedClient(sources: []),
            repository: repository,
            sources: []
        )
        XCTAssertEqual(reloadedStore.latestBriefingSnapshot, first)
    }

    func testShareUsesSavedContentAndPersistsMetadata() throws {
        let repository = FeedRepository(inMemory: true)
        let result = try XCTUnwrap(FounderModePreviewFixtures.freshFeed.result)
        let store = makeStore(repository: repository, result: result)
        let snapshot = try XCTUnwrap(store.saveCurrentBriefingSnapshot(savedAt: savedAt))
        let sharedAt = savedAt.addingTimeInterval(120)
        let savedTitle = try XCTUnwrap(snapshot.items.compactMap(\.article).first?.article.title)

        store.recordBriefingShare(id: snapshot.id, at: sharedAt)

        let persisted = try XCTUnwrap(repository.loadBriefingSnapshots().first)
        XCTAssertEqual(persisted.shareCount, 1)
        XCTAssertEqual(persisted.lastSharedAt, sharedAt)
        XCTAssertTrue(persisted.exportText.contains("Signal Feed Founder Briefing"))
        XCTAssertTrue(persisted.exportText.contains("Intent: Build Controls AI"))
        XCTAssertTrue(persisted.exportText.contains("Reading budget: 20 minutes"))
        XCTAssertTrue(persisted.exportText.contains(savedTitle))
    }

    func testCompletionStoresIntentAndStopTimeOnceAcrossRelaunch() throws {
        let repository = FeedRepository(inMemory: true)
        let result = try XCTUnwrap(FounderModePreviewFixtures.freshFeed.result)
        let store = makeStore(repository: repository, result: result)
        let completedAt = savedAt.addingTimeInterval(300)

        let first = try XCTUnwrap(store.completeCurrentBriefing(at: completedAt))
        let duplicate = try XCTUnwrap(store.completeCurrentBriefing(at: completedAt.addingTimeInterval(60)))

        XCTAssertEqual(first.id, duplicate.id)
        XCTAssertEqual(store.briefingCompletions.count, 1)
        XCTAssertEqual(first.completedAt, completedAt)
        XCTAssertEqual(first.intent, .buildControlsAI)
        XCTAssertEqual(first.readingBudgetMinutes, 20)
        XCTAssertEqual(first.reviewedSlotCount, 5)

        let reloaded = FeedStore(client: FeedClient(sources: []), repository: repository, sources: [])
        XCTAssertEqual(reloaded.briefingCompletions, [first])
    }

    func testOpportunityHypothesisIncludesRequiredFieldsAndResearchQueuePersists() throws {
        let repository = FeedRepository(inMemory: true)
        let result = try XCTUnwrap(FounderModePreviewFixtures.freshFeed.result)
        let flaggedArticle = try XCTUnwrap(result.articles.first { !$0.opportunityLabels.isEmpty })
        let store = makeStore(repository: repository, result: result)

        store.syncOpportunityHypotheses(from: [flaggedArticle], at: savedAt)

        let hypothesis = try XCTUnwrap(store.opportunityHypotheses.first)
        XCTAssertFalse(hypothesis.customer.isEmpty)
        XCTAssertFalse(hypothesis.trigger.isEmpty)
        XCTAssertFalse(hypothesis.wedge.isEmpty)
        XCTAssertFalse(hypothesis.evidence.isEmpty)
        XCTAssertFalse(hypothesis.nextTest.isEmpty)
        XCTAssertEqual(hypothesis.supportingArticle.id, flaggedArticle.id)
        XCTAssertEqual(hypothesis.confidence, flaggedArticle.decisionSummary.confidence)

        store.toggleResearchQueue(id: hypothesis.id)

        let reloaded = FeedStore(client: FeedClient(sources: []), repository: repository, sources: [])
        let persisted = try XCTUnwrap(reloaded.opportunityHypotheses.first { $0.id == hypothesis.id })
        XCTAssertTrue(persisted.isSavedToResearchQueue)
        XCTAssertEqual(reloaded.savedResearchHypotheses.map(\.id), [hypothesis.id])
    }

    private func makeStore(repository: FeedRepository, result: DigestResult) -> FeedStore {
        let store = FeedStore(
            client: FeedClient(sources: []),
            repository: repository,
            sources: []
        )
        store.state = .loaded(result)
        return store
    }
}
