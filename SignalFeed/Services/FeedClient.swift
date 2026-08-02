import Foundation

struct FeedClient {
    private let defaultSources: [FeedSource]

    init(sources: [FeedSource] = DefaultContentConfig.sources) {
        self.defaultSources = sources
    }

    func fetchDigest(
        settings: FeedSettings,
        intent: IntentMode = .buildControlsAI,
        preferenceProfile: FounderPreferenceProfile = .industrialAI,
        learningProfile: RankingLearningProfile = .empty
    ) async -> DigestResult {
        await fetchDigest(sources: defaultSources, settings: settings, intent: intent, preferenceProfile: preferenceProfile, learningProfile: learningProfile)
    }

    func fetchDigest(
        sources: [FeedSource],
        settings: FeedSettings,
        intent: IntentMode = .buildControlsAI,
        preferenceProfile: FounderPreferenceProfile = .industrialAI,
        learningProfile: RankingLearningProfile = .empty
    ) async -> DigestResult {
        var allArticles: [FeedArticle] = []
        var sourceHealth: [SourceHealth] = []

        await withTaskGroup(of: SourceFetchResult.self) { group in
            for source in sources {
                group.addTask {
                    await Self.fetch(source: source)
                }
            }

            for await result in group {
                allArticles.append(contentsOf: result.articles)
                sourceHealth.append(result.health)
            }
        }

        let rankableArticles: [FeedArticle]
        if settings.isFullTextExtractionEnabled {
            rankableArticles = await FullTextExtractor.enrich(
                articles: allArticles,
                limit: settings.fullTextArticleLimit
            )
        } else {
            rankableArticles = allArticles
        }

        let rankedArticles = RankingEngine.rank(
            articles: rankableArticles,
            settings: settings,
            intent: intent,
            preferenceProfile: preferenceProfile,
            learningProfile: learningProfile
        )

        return DigestResult(
            articles: rankedArticles,
            sourceHealth: sourceHealth.sorted { $0.source.name < $1.source.name },
            generatedAt: Date(),
            fetchedCount: allArticles.count
        )
    }

    private static func fetch(source: FeedSource) async -> SourceFetchResult {
        let start = Date()

        do {
            guard let url = URL(string: source.url) else {
                throw FeedClientError.invalidURL
            }

            var request = URLRequest(url: url)
            request.timeoutInterval = 18
            request.cachePolicy = .reloadRevalidatingCacheData
            request.setValue("SignalFeed-iOS/1.0", forHTTPHeaderField: "User-Agent")

            let (data, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse,
               !(200..<300).contains(httpResponse.statusCode) {
                throw FeedClientError.httpStatus(httpResponse.statusCode)
            }

            let articles = RSSFeedParser.parse(data: data, source: source)
            let status: SourceStatus = articles.isEmpty ? .empty : .healthy

            return SourceFetchResult(
                articles: articles,
                health: SourceHealth(
                    source: source,
                    status: status,
                    itemCount: articles.count,
                    message: articles.isEmpty ? "Parsed feed, but no entries were found." : "Fetched and parsed successfully.",
                    duration: Date().timeIntervalSince(start)
                )
            )
        } catch {
            return SourceFetchResult(
                articles: [],
                health: SourceHealth(
                    source: source,
                    status: .failed,
                    itemCount: 0,
                    message: String(describing: error),
                    duration: Date().timeIntervalSince(start)
                )
            )
        }
    }
}

private struct SourceFetchResult: Sendable {
    let articles: [FeedArticle]
    let health: SourceHealth
}

private enum FeedClientError: Error {
    case invalidURL
    case httpStatus(Int)
}
