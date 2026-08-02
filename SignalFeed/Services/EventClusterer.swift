import Foundation

enum EventClusterer {
    static func cluster(_ articles: [ScoredArticle]) -> [ScoredArticle] {
        let ordered = articles.sorted { stableKey(for: $0) < stableKey(for: $1) }
        guard ordered.count > 1 else { return ordered }

        var parents = Array(ordered.indices)

        func root(of index: Int) -> Int {
            var current = index
            while parents[current] != current {
                current = parents[current]
            }
            return current
        }

        func unite(_ first: Int, _ second: Int) {
            let firstRoot = root(of: first)
            let secondRoot = root(of: second)
            guard firstRoot != secondRoot else { return }
            let lower = min(firstRoot, secondRoot)
            let upper = max(firstRoot, secondRoot)
            parents[upper] = lower
        }

        for first in ordered.indices {
            for second in ordered.indices where second > first {
                if representsSameEvent(ordered[first], ordered[second]) {
                    unite(first, second)
                }
            }
        }

        let grouped = Dictionary(grouping: ordered.indices, by: { root(of: $0) })
        return grouped.keys.sorted().compactMap { groupID in
            let members = grouped[groupID, default: []].map { ordered[$0] }
            guard let representative = members.sorted(by: representativeComesFirst).first else { return nil }
            let alternates = members
                .filter { $0.id != representative.id }
                .sorted(by: representativeComesFirst)
                .map(\.article)

            guard !alternates.isEmpty else { return representative }
            return attachingCluster(
                ArticleEventCluster(
                    id: "event:" + members.map(\.id).sorted().joined(separator: "|"),
                    alternateArticles: alternates
                ),
                to: representative
            )
        }
    }

    private static func representsSameEvent(_ first: ScoredArticle, _ second: ScoredArticle) -> Bool {
        let titleSimilarity = jaccard(tokens(first.article.title), tokens(second.article.title))
        if titleSimilarity >= 0.50 {
            return true
        }

        let firstContent = tokens(first.article.title + " " + first.article.summary)
        let secondContent = tokens(second.article.title + " " + second.article.summary)
        return titleSimilarity >= 0.30 && jaccard(firstContent, secondContent) >= 0.55
    }

    private static func representativeComesFirst(_ first: ScoredArticle, _ second: ScoredArticle) -> Bool {
        if first.article.sourceReputation != second.article.sourceReputation {
            return first.article.sourceReputation > second.article.sourceReputation
        }
        if first.score != second.score {
            return first.score > second.score
        }
        let firstDate = first.article.publishedAt ?? .distantPast
        let secondDate = second.article.publishedAt ?? .distantPast
        if firstDate != secondDate {
            return firstDate > secondDate
        }
        return stableKey(for: first) < stableKey(for: second)
    }

    private static func attachingCluster(_ cluster: ArticleEventCluster, to article: ScoredArticle) -> ScoredArticle {
        let clusterReason = RankingReason(
            kind: .duplicate,
            title: "Event cluster",
            detail: "Selected \(article.article.sourceName) as representative for \(cluster.sourceCount) related sources.",
            impact: 0
        )
        let clusterPenalty = ScorePenalty(
            title: "Duplicate topic",
            value: 0,
            explanation: "Related coverage was consolidated under the highest-reputation source."
        )

        return ScoredArticle(
            article: article.article,
            score: article.score,
            scoreComponents: article.scoreComponents,
            penalties: article.penalties + [clusterPenalty],
            decisionSummary: article.decisionSummary,
            category: article.category,
            reasons: article.reasons,
            reasonDetails: article.reasonDetails + [clusterReason],
            matchedKeywords: article.matchedKeywords,
            opportunityLabels: article.opportunityLabels,
            canonicalURL: article.canonicalURL,
            eventCluster: cluster
        )
    }

    private static func stableKey(for article: ScoredArticle) -> String {
        article.canonicalURL.isEmpty ? article.id : article.canonicalURL
    }

    private static func tokens(_ text: String) -> Set<String> {
        Set(
            text.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count > 2 && !stopWords.contains($0) }
        )
    }

    private static func jaccard(_ first: Set<String>, _ second: Set<String>) -> Double {
        guard !first.isEmpty, !second.isEmpty else { return 0 }
        return Double(first.intersection(second).count) / Double(first.union(second).count)
    }

    private static let stopWords: Set<String> = [
        "and", "are", "for", "from", "has", "into", "its", "new", "the", "this", "that", "with"
    ]
}
