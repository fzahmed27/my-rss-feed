import Foundation

@MainActor
enum FounderModePreviewFixtures {
    static let referenceDate = Date(timeIntervalSince1970: 1_785_500_000)

    static var freshFeed: FeedStore {
        makeStore(articles: rankedArticles)
    }

    static var caughtUpFeed: FeedStore {
        let articles = rankedArticles
        let interactions = Dictionary(uniqueKeysWithValues: articles.prefix(15).map { article in
            (
                article.id,
                ArticleInteraction(
                    isBookmarked: false,
                    feedback: nil,
                    isRead: true,
                    updatedAt: referenceDate
                )
            )
        })
        return makeStore(articles: articles, interactions: interactions)
    }

    static var noHighQualityFeed: FeedStore {
        makeStore(articles: [], fetchedCount: 24)
    }

    static var mutedTopicFeed: FeedStore {
        let store = makeStore(
            articles: rankedArticles.filter { !$0.article.searchableText.contains("plc") }
        )
        let topic = MutedTopic(
            topic: "PLC",
            expiresAt: referenceDate.addingTimeInterval(30 * 86_400),
            createdAt: referenceDate
        )
        store.mutedTopics = [topic.id: topic]
        return store
    }

    static var longTitleArticle: ScoredArticle {
        let article = FeedArticle(
            id: "long-title",
            sourceID: "automation-world",
            sourceName: "Automation World",
            sourceReputation: 1.2,
            sourceKind: .feed,
            title: "A deeply technical look at deterministic edge inference for safety-critical PLC control loops across brownfield manufacturing systems",
            link: "https://example.com/long-title",
            summary: "A deployment study connects embedded AI, controls reliability, and practical modernization constraints.",
            content: "The study evaluates edge inference for industrial control systems and predictive maintenance.",
            extractedContent: String(repeating: "Industrial controls evidence and implementation detail. ", count: 60),
            publishedAt: referenceDate.addingTimeInterval(-3_600)
        )
        return rank([article]).first ?? .placeholder
    }

    private static let sources = [
        FeedSource(id: "automation-world", name: "Automation World", url: "https://example.com/automation", reputation: 1.2),
        FeedSource(id: "primary-research", name: "Primary Research Lab", url: "https://example.com/research", reputation: 1.25),
        FeedSource(id: "founder-signals", name: "Founder Signals", url: "https://example.com/startups", reputation: 1.1)
    ]

    private static var rankedArticles: [ScoredArticle] {
        rank(stories.enumerated().map { index, story in
            let source = sources[index % sources.count]
            return FeedArticle(
                id: "founder-preview-\(index)",
                sourceID: source.id,
                sourceName: source.name,
                sourceReputation: source.reputation,
                sourceKind: source.kind,
                title: story.title,
                link: "https://example.com/founder-preview-\(index)",
                summary: story.summary,
                content: story.summary,
                extractedContent: String(repeating: "\(story.summary) ", count: 35),
                publishedAt: referenceDate.addingTimeInterval(TimeInterval(-index * 3_600))
            )
        })
    }

    private static func rank(_ articles: [FeedArticle]) -> [ScoredArticle] {
        RankingEngine.rank(
            articles: articles,
            settings: FeedSettings(
                minimumScore: 0,
                daysBack: 90,
                maxArticles: 100,
                isFullTextExtractionEnabled: false,
                fullTextArticleLimit: 0
            ),
            intent: .buildControlsAI,
            now: referenceDate
        )
    }

    private static func makeStore(
        articles: [ScoredArticle],
        interactions: [String: ArticleInteraction] = [:],
        fetchedCount: Int? = nil
    ) -> FeedStore {
        let store = FeedStore(
            client: FeedClient(sources: []),
            repository: FeedRepository(inMemory: true),
            sources: sources
        )
        store.interactions = interactions
        store.savedArticles = Dictionary(uniqueKeysWithValues: articles.map { ($0.id, $0) })
        store.sourceMetrics = Dictionary(uniqueKeysWithValues: sources.map { source in
            (
                source.id,
                SourceHealthSummary(
                    lastFetchAt: referenceDate,
                    lastSuccessfulFetchAt: referenceDate,
                    failureStreak: 0,
                    totalFetches: 12,
                    averageQuality: 78,
                    lastStatus: .healthy,
                    lastMessage: "Fixture data loaded.",
                    lastItemCount: articles.count,
                    lastDuration: 0.2
                )
            )
        })
        store.state = .loaded(
            DigestResult(
                articles: articles,
                sourceHealth: sources.map { source in
                    SourceHealth(
                        source: source,
                        status: .healthy,
                        itemCount: articles.filter { $0.article.sourceID == source.id }.count,
                        message: "Fixture data loaded.",
                        duration: 0.2
                    )
                },
                generatedAt: referenceDate,
                fetchedCount: fetchedCount ?? articles.count
            )
        )
        return store
    }

    private static let stories: [(title: String, summary: String)] = [
        ("PLC vendors add deterministic edge inference", "Industrial controllers can now run anomaly detection beside control logic."),
        ("PID tuning assistant cuts commissioning time", "A controls workflow turns plant response data into safer tuning recommendations."),
        ("Factory sensor fusion improves predictive maintenance", "Vibration and current signals expose motor faults before downtime."),
        ("Embedded AI runtime targets industrial gateways", "The runtime reduces memory use for offline edge inference."),
        ("Robotic workcell learns contact-rich assembly", "A manufacturing robot adapts insertion force using tactile feedback."),
        ("Developer toolkit traces AI decisions on device", "New diagnostics help teams inspect model behavior at the edge."),
        ("Industrial startup wins first multi-site rollout", "A pilot converted into a commercial deployment across six plants."),
        ("Signal processing library adds streaming transforms", "The update improves real-time feature extraction from machine sensors."),
        ("Machine builder standardizes controls telemetry", "A common event schema makes equipment data easier to operationalize."),
        ("Vision inspection model runs without cloud access", "A compact model detects defects within factory latency budgets."),
        ("Primary study benchmarks safe robot manipulation", "Researchers publish failure modes relevant to industrial deployment."),
        ("Automation distributor launches edge AI practice", "A channel partner begins packaging deployment and support services."),
        ("Open-source simulator models motor-control faults", "Controls teams can test detection logic before connecting hardware."),
        ("Industrial cybersecurity rule changes buying criteria", "Manufacturers will require stronger evidence from connected equipment vendors."),
        ("Semiconductor update lowers inference power draw", "A new accelerator profile improves always-on sensor processing."),
        ("Manufacturing software firm expands usage pricing", "The commercial model ties automation software cost to deployed assets."),
        ("Space robotics mission validates autonomous repair", "On-orbit manipulation offers lessons for robust remote operation."),
        ("AI coding agent adds embedded C diagnostics", "The developer tool can explain memory and timing failures in firmware."),
        ("Digital twin standard gains controls interoperability", "New interfaces connect simulation results with PLC engineering workflows."),
        ("Battery plant uses acoustic sensing for quality", "Inline signal analysis catches process drift before final inspection.")
    ]
}
