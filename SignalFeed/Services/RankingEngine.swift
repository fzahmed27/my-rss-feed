import Foundation

enum RankingEngine {
    static func rank(
        articles: [FeedArticle],
        settings: FeedSettings,
        intent: IntentMode = .buildControlsAI,
        keywords: [String: Double] = DefaultContentConfig.keywords,
        preferenceProfile: FounderPreferenceProfile = .industrialAI,
        learningProfile: RankingLearningProfile = .empty,
        now: Date = Date()
    ) -> [ScoredArticle] {
        let cutoff = Calendar.current.date(byAdding: .day, value: -settings.daysBack, to: now)
        let mutedTopics = learningProfile.mutedTopics.filter { $0.value > now }.keys

        let initiallyScored = articles.compactMap { article -> ScoredArticle? in
            if let cutoff, let publishedAt = article.publishedAt, publishedAt < cutoff {
                return nil
            }
            if mutedTopics.contains(where: { article.searchableText.contains($0) }) {
                return nil
            }

            let scoredArticle = score(
                article: article,
                intent: intent,
                keywords: keywords,
                preferenceProfile: preferenceProfile,
                learningProfile: learningProfile,
                now: now
            )
            guard scoredArticle.score >= settings.minimumScore else { return nil }
            return scoredArticle
        }

        let uniqueArticles = removeExactURLDuplicates(initiallyScored)
        let scored = EventClusterer.cluster(uniqueArticles)
            .filter { $0.score >= settings.minimumScore }
            .sorted { first, second in
                if first.score == second.score {
                    return (first.article.publishedAt ?? .distantPast) > (second.article.publishedAt ?? .distantPast)
                }
                return first.score > second.score
            }

        return Array(scored.prefix(settings.maxArticles))
    }

    static func canonicalURL(from rawURL: String) -> String {
        guard var components = URLComponents(string: rawURL.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            return rawURL.lowercased()
        }

        components.scheme = components.scheme?.lowercased()
        components.host = components.host?.lowercased()
        components.fragment = nil

        let blockedQueryPrefixes = ["utm_"]
        let blockedQueryNames = Set(["fbclid", "gclid", "mc_cid", "mc_eid"])
        components.queryItems = components.queryItems?.filter { item in
            !blockedQueryNames.contains(item.name.lowercased())
                && !blockedQueryPrefixes.contains { item.name.lowercased().hasPrefix($0) }
        }

        if components.path.hasSuffix("/"), components.path.count > 1 {
            components.path.removeLast()
        }

        return components.string ?? rawURL.lowercased()
    }

    private static func score(
        article: FeedArticle,
        intent: IntentMode,
        keywords: [String: Double],
        preferenceProfile: FounderPreferenceProfile,
        learningProfile: RankingLearningProfile,
        now: Date
    ) -> ScoredArticle {
        let text = article.searchableText
        let title = article.title.lowercased()
        let body = [article.summary, article.content, article.extractedContent].joined(separator: " ").lowercased()
        let category = categorize(article: article)
        let opportunityLabels = OpportunityDetector.labels(for: article)

        let keywordMatches = weightedMatches(in: text, weights: keywords)
        let preferenceMatches = weightedMatches(in: text, weights: preferenceProfile.topicWeights)
        let intentMatches = weightedMatches(in: text, weights: preferenceProfile.intentWeights[intent] ?? [:])
        let penaltyMatches = weightedMatches(in: text, weights: preferenceProfile.penaltyWeights)
        let allMatchedKeywords = Set(keywordMatches.map(\.term))
            .union(preferenceMatches.map(\.term))
            .union(intentMatches.map(\.term))

        let learningBoost = allMatchedKeywords.reduce(0.0) { total, term in
            total
                + learningProfile.keywordAdjustments[term, default: 0]
                + learningProfile.topicAdjustments[term, default: 0]
        }
        let sourceAdjustment = learningProfile.sourceAdjustments[article.sourceID, default: 0]

        let personalRaw = clamp01(
            (preferenceMatches.reduce(0.0) { $0 + $1.weight } / 4.5)
                + (intentMatches.reduce(0.0) { $0 + $1.weight } / 4.0)
                + (learningBoost / 10.0)
        )
        let actionabilityRaw = clamp01(
            matchCount(in: text, terms: actionTerms) / 5.0
                + Double(opportunityLabels.count) * 0.16
                + (containsAny(text, ["rfp", "procurement", "customer", "pilot"]) ? 0.18 : 0)
        )
        let strategicRaw = clamp01(
            matchCount(in: text, terms: strategicTerms) / 5.0
                + (category == .markets || category == .startups ? 0.18 : 0)
                + Double(opportunityLabels.count) * 0.12
        )
        let sourceRaw = clamp01((article.sourceReputation - 0.75) / 0.65 + sourceAdjustment / 12.0)
        let noveltyRaw = clamp01(
            0.7
                + (article.hasExtractedContent ? 0.12 : 0)
                - penaltyMatches.reduce(0.0) { $0 + $1.weight } / 5.0
                - (containsAny(title, genericAnnouncementTerms) ? 0.18 : 0)
        )
        let recencyRaw = recencyValue(publishedAt: article.publishedAt, now: now)

        let componentInputs: [(ScoreComponentKind, Double, String)] = [
            (.personalRelevance, personalRaw, explanation(for: preferenceMatches, intentMatches: intentMatches, fallback: "Matches the selected founder intent and preference profile.")),
            (.actionability, actionabilityRaw, actionabilityRaw > 0.55 ? "Contains concrete business, technical, or product follow-up signals." : "Limited immediate action signal."),
            (.strategicImportance, strategicRaw, strategicRaw > 0.55 ? "Could affect market timing, roadmap, or industrial AI positioning." : "Mostly background context."),
            (.sourceQuality, sourceRaw, "\(article.sourceName) reputation is \(article.sourceReputation.formatted(.number.precision(.fractionLength(2))))."),
            (.novelty, noveltyRaw, noveltyRaw > 0.65 ? "Appears distinct enough to review." : "Likely repetitive or shallow coverage."),
            (.recency, recencyRaw, recencyRaw > 0.7 ? "Recent enough to affect near-term decisions." : "Older item; ranking depends on enduring value.")
        ]

        let components = componentInputs.map { kind, rawValue, explanation in
            let normalized = clamp01(rawValue)
            let weight = preferenceProfile.componentWeights[kind, default: 0]
            return ScoreComponent(
                kind: kind,
                rawValue: rawValue,
                normalizedValue: normalized,
                weight: weight,
                contribution: normalized * weight * 100,
                explanation: explanation
            )
        }

        var penalties = explicitPenalties(
            text: text,
            title: title,
            body: body,
            penaltyMatches: penaltyMatches,
            publishedAt: article.publishedAt,
            now: now
        )
        let penaltyTotal = penalties.reduce(0.0) { $0 + $1.value }
        let baseScore = components.reduce(0.0) { $0 + $1.contribution }
        let score = max(0, baseScore - penaltyTotal)
        let confidence = signalConfidence(
            article: article,
            sourceQuality: sourceRaw,
            novelty: noveltyRaw,
            penaltyTotal: penaltyTotal
        )

        var reasonDetails = components.map { component in
            RankingReason(
                kind: reasonKind(for: component.kind),
                title: component.kind.title,
                detail: component.explanation,
                impact: component.contribution
            )
        }
        reasonDetails.append(contentsOf: penalties.map { penalty in
            RankingReason(kind: .duplicate, title: penalty.title, detail: penalty.explanation, impact: -penalty.value)
        })
        if article.hasExtractedContent {
            reasonDetails.append(
                RankingReason(
                    kind: .fullText,
                    title: "Full text analyzed",
                    detail: "Used extracted page text for ranking and explanation.",
                    impact: 0
                )
            )
        }

        let topComponent = components.max { $0.contribution < $1.contribution }
        let visibleReasons = [
            topComponent.map { "\($0.kind.title): \($0.explanation)" },
            opportunityLabels.first.map { "Opportunity flag: \($0.title)" }
        ].compactMap { $0 }

        if penalties.isEmpty {
            penalties.append(ScorePenalty(title: "Duplicate penalty: none", value: 0, explanation: "No duplicate topic penalty applied."))
        }

        return ScoredArticle(
            article: article,
            score: score,
            scoreComponents: components,
            penalties: penalties,
            decisionSummary: decisionSummary(
                article: article,
                category: category,
                matchedTerms: Array(allMatchedKeywords).sorted(),
                opportunityLabels: opportunityLabels,
                actionability: actionabilityRaw,
                strategicImportance: strategicRaw,
                confidence: confidence
            ),
            category: category,
            reasons: visibleReasons.isEmpty ? ["Why this matters: matches your configured sources."] : visibleReasons,
            reasonDetails: reasonDetails,
            matchedKeywords: Array(allMatchedKeywords).sorted(),
            opportunityLabels: opportunityLabels,
            canonicalURL: canonicalURL(from: article.link)
        )
    }

    private static func removeExactURLDuplicates(_ articles: [ScoredArticle]) -> [ScoredArticle] {
        var seenURLs = Set<String>()
        var deduped: [ScoredArticle] = []

        for article in articles {
            let canonical = article.canonicalURL
            guard canonical.isEmpty || !seenURLs.contains(canonical) else { continue }
            if !canonical.isEmpty {
                seenURLs.insert(canonical)
            }
            deduped.append(article)
        }

        return deduped
    }

    private static func weightedMatches(in text: String, weights: [String: Double]) -> [(term: String, weight: Double)] {
        weights
            .map { ($0.key.lowercased(), $0.value) }
            .filter { text.contains($0.0) }
            .sorted { first, second in
                if first.1 == second.1 { return first.0 < second.0 }
                return first.1 > second.1
            }
    }

    private static func explicitPenalties(
        text: String,
        title: String,
        body: String,
        penaltyMatches: [(term: String, weight: Double)],
        publishedAt: Date?,
        now: Date
    ) -> [ScorePenalty] {
        var penalties: [ScorePenalty] = []

        for match in penaltyMatches.prefix(3) {
            penalties.append(
                ScorePenalty(
                    title: penaltyTitle(for: match.term),
                    value: match.weight * 7,
                    explanation: "Matched low-signal pattern: \(match.term)."
                )
            )
        }

        if containsAny(title, genericAnnouncementTerms), !containsAny(text, industrialTerms) {
            penalties.append(ScorePenalty(title: "Generic announcement", value: 8, explanation: "Announcement language lacks industrial, technical, or commercial specificity."))
        }

        if body.count < 500, !containsAny(text, actionTerms) {
            penalties.append(ScorePenalty(title: "Low technical depth", value: 6, explanation: "The available content is short and lacks practical detail."))
        }

        if let publishedAt, now.timeIntervalSince(publishedAt) > 14 * 86_400, !containsAny(text, enduringTerms) {
            penalties.append(ScorePenalty(title: "Old low-enduring content", value: 5, explanation: "Older item without durable reference value."))
        }

        return penalties
    }

    private static func decisionSummary(
        article: FeedArticle,
        category: ArticleCategory,
        matchedTerms: [String],
        opportunityLabels: [OpportunityLabel],
        actionability: Double,
        strategicImportance: Double
        ,
        confidence: SignalConfidence
    ) -> ArticleDecisionSummary {
        let leadingTerm = matchedTerms.first ?? category.title.lowercased()
        let whatChanged = conciseChange(from: article)
        let whyItMatters: String
        if matchedTerms.contains(where: { industrialTerms.contains($0) }) {
            whyItMatters = "It maps directly to industrial AI, controls, robotics, or edge deployment work."
        } else if !opportunityLabels.isEmpty {
            whyItMatters = "It may signal a commercial, market, or product-timing shift worth tracking."
        } else {
            whyItMatters = "It adds context for your \(leadingTerm) watchlist."
        }

        let action: String
        if actionability > 0.65 {
            action = "Save it, extract the concrete next step, or add it to a current project note."
        } else if strategicImportance > 0.65 {
            action = "Skim for roadmap implications, then save only if it changes priorities."
        } else {
            action = "Skim briefly or dismiss if it repeats what you already know."
        }

        let shouldCare: String
        if actionability > 0.65 || strategicImportance > 0.7 {
            shouldCare = "Yes: it has a plausible product, market, or roadmap implication."
        } else if matchedTerms.contains(where: { industrialTerms.contains($0) }) || !opportunityLabels.isEmpty {
            shouldCare = "Maybe: it matches your profile, but the immediate implication is still light."
        } else {
            shouldCare = "No: treat it as background unless it connects to a current question."
        }

        return ArticleDecisionSummary(
            whyThisMatters: "Why this matters: \(whyItMatters)",
            whatChanged: whatChanged,
            whyItMattersToYou: whyItMatters,
            shouldCare: shouldCare,
            suggestedAction: action,
            confidence: confidence,
            primaryUncertainty: primaryUncertainty(for: article, confidence: confidence),
            estimatedReadingMinutes: estimatedReadingMinutes(for: article),
            evidence: evidenceSummary(for: article)
        )
    }

    private static func signalConfidence(
        article: FeedArticle,
        sourceQuality: Double,
        novelty: Double,
        penaltyTotal: Double
    ) -> SignalConfidence {
        if article.hasExtractedContent, sourceQuality > 0.62, novelty > 0.55, penaltyTotal < 8 {
            return .high
        }
        if sourceQuality > 0.35, penaltyTotal < 18 {
            return .medium
        }
        return .low
    }

    private static func primaryUncertainty(for article: FeedArticle, confidence: SignalConfidence) -> String {
        switch confidence {
        case .high:
            return "Ranking is based on source metadata plus extracted article text."
        case .medium:
            return article.hasExtractedContent
                ? "The implication is inferred from article text; corroboration may still be limited."
                : "Full-text extraction was unavailable, so the implication leans on feed metadata."
        case .low:
            return "Weak source quality, sparse text, age, or low-signal patterns reduce confidence."
        }
    }

    private static func evidenceSummary(for article: FeedArticle) -> String {
        if let publishedAt = article.publishedAt {
            return "Source: \(article.sourceName), published \(publishedAt.formatted(date: .abbreviated, time: .omitted))."
        }
        return "Source: \(article.sourceName)."
    }

    private static func conciseChange(from article: FeedArticle) -> String {
        let sourceText = article.summary.isEmpty ? article.title : article.summary
        let sentence = sourceText
            .split(whereSeparator: { ".!?".contains($0) })
            .first
            .map(String.init) ?? article.title
        return sentence.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func estimatedReadingMinutes(for article: FeedArticle) -> Int {
        let words = article.rankingContent.split { $0.isWhitespace || $0.isNewline }.count
        return max(1, Int(ceil(Double(max(words, 180)) / 220.0)))
    }

    private static func explanation(
        for preferenceMatches: [(term: String, weight: Double)],
        intentMatches: [(term: String, weight: Double)],
        fallback: String
    ) -> String {
        let terms = Array(Set((preferenceMatches + intentMatches).prefix(3).map(\.term))).sorted()
        guard !terms.isEmpty else { return fallback }
        return "Matches \(terms.joined(separator: ", "))."
    }

    private static func recencyValue(publishedAt: Date?, now: Date) -> Double {
        guard let publishedAt else { return 0.35 }
        let age = max(0, now.timeIntervalSince(publishedAt))
        switch age {
        case 0..<(36 * 60 * 60): return 1.0
        case 0..<(7 * 86_400): return 0.72
        case 0..<(30 * 86_400): return 0.38
        default: return 0.12
        }
    }

    private static func reasonKind(for component: ScoreComponentKind) -> RankingReasonKind {
        switch component {
        case .personalRelevance: .personal
        case .actionability: .opportunity
        case .strategicImportance: .keyword
        case .sourceQuality: .sourceTrust
        case .novelty: .duplicate
        case .recency: .freshness
        }
    }

    private static func categorize(article: FeedArticle) -> ArticleCategory {
        if article.sourceKind == .person {
            return .people
        }

        let text = article.searchableText
        let categories: [(ArticleCategory, [String])] = [
            (.tactile, ["tactile", "haptic", "haptics", "touch sensing", "force sensing"]),
            (.vision, ["computer vision", "image recognition", "object detection", "semantic segmentation", "depth estimation", "cnn", "convolutional"]),
            (.robotics, ["robotics", "robot", "grasp", "grasping", "manipulation", "dexterous", "autonomous", "manufacturing robotics"]),
            (.ai, ["machine learning", "deep learning", "neural network", "pytorch", "ai agents", "foundation model", "llm", "embedded ai"]),
            (.hardware, ["sensor", "sensors", "embedded", "edge", "inference", "model compression", "quantization", "optimization", "gpu", "chip", "plc"]),
            (.markets, ["economics", "economy", "inflation", "monetary policy", "fiscal policy", "gdp", "recession", "stock", "investing", "portfolio", "trading", "equity", "bonds", "etf", "valuation", "dividend", "guidance", "earnings"]),
            (.startups, ["startup", "funding", "seed round", "series a", "venture", "customer", "enterprise adoption", "commercialization", "go-to-market"])
        ]

        return categories
            .map { category, terms in (category, terms.filter { text.contains($0) }.count) }
            .max { $0.1 < $1.1 }
            .flatMap { $0.1 > 0 ? $0.0 : nil } ?? .other
    }

    private static func matchCount(in text: String, terms: [String]) -> Double {
        Double(terms.filter { text.contains($0) }.count)
    }

    private static func containsAny(_ text: String, _ patterns: [String]) -> Bool {
        patterns.contains { text.contains($0) }
    }

    private static func clamp01(_ value: Double) -> Double {
        max(0, min(1, value))
    }

    private static func penaltyTitle(for term: String) -> String {
        switch term {
        case "funding": "Funding without relevance"
        case "model launch": "Repetitive model-launch coverage"
        default: "Previously ignored theme"
        }
    }

    private static let industrialTerms = [
        "industrial automation", "plc", "plcs", "pid tuning", "control systems",
        "embedded ai", "edge inference", "robotics for manufacturing", "manufacturing robotics",
        "sensor", "sensors", "signal processing", "predictive maintenance", "factory", "edge ai"
    ]
    private static let actionTerms = [
        "rfp", "request for proposal", "procurement", "pilot", "customer", "launched",
        "released", "open source", "api", "benchmark", "grant", "compliance", "deployment"
    ]
    private static let strategicTerms = [
        "regulation", "mandate", "enterprise adoption", "earnings", "guidance", "sec filing",
        "10-k", "10-q", "8-k", "funding", "commercialization", "go-to-market", "rate cut", "cpi"
    ]
    private static let genericAnnouncementTerms = [
        "announces", "announcement", "unveils", "introduces", "model launch", "funding round"
    ]
    private static let enduringTerms = [
        "tutorial", "reference", "architecture", "research", "benchmark", "control systems", "sensor", "plc"
    ]
}

enum OpportunityDetector {
    static func labels(for article: FeedArticle) -> [OpportunityLabel] {
        let text = article.searchableText
        var labels: [OpportunityLabel] = []

        if containsAny(text, [
            "earnings", "guidance", "sec filing", "10-k", "10-q", "8-k",
            "fomc", "cpi", "jobs report", "inflation print", "rate cut", "rate hike"
        ]) {
            labels.append(.marketMoving)
        }

        if containsAny(text, [
            "rfp", "request for proposal", "grant", "procurement", "regulation",
            "compliance", "new requirement", "mandate", "enterprise adoption", "pilot"
        ]) {
            labels.append(.businessOpportunity)
        }

        if containsAny(text, [
            "launches", "released", "open source", "model release", "api",
            "agent", "agents", "tooling", "benchmark", "developer tools"
        ]) {
            labels.append(.aiLaunch)
        }

        return labels
    }

    private static func containsAny(_ text: String, _ patterns: [String]) -> Bool {
        patterns.contains { text.contains($0) }
    }

    static func detail(for label: OpportunityLabel) -> String {
        switch label {
        case .marketMoving:
            "Matched language around earnings, guidance, SEC filings, macro prints, or rates."
        case .businessOpportunity:
            "Matched language around RFPs, grants, regulation, compliance, procurement, or adoption shifts."
        case .aiLaunch:
            "Matched language around launches, model releases, open-source tooling, APIs, agents, or benchmarks."
        }
    }
}
