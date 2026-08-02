import Foundation

struct OpportunityHypothesis: Identifiable, Codable, Hashable, Sendable {
    let id: String
    let label: OpportunityLabel
    let problem: String
    let customer: String
    let trigger: String
    let wedge: String
    let evidence: String
    let confidence: SignalConfidence
    let nextTest: String
    let supportingArticle: ScoredArticle
    let createdAt: Date
    var isSavedToResearchQueue: Bool
}

enum OpportunityHypothesisGenerator {
    static func generate(
        from articles: [ScoredArticle],
        createdAt: Date = Date()
    ) -> [OpportunityHypothesis] {
        articles.compactMap { article in
            guard let label = article.opportunityLabels.first else { return nil }
            let text = article.article.searchableText

            return OpportunityHypothesis(
                id: "hypothesis:\(article.id):\(label.rawValue)",
                label: label,
                problem: article.decisionSummary.whyItMattersToYou,
                customer: customer(for: text, label: label),
                trigger: trigger(for: article, label: label),
                wedge: wedge(for: article, text: text),
                evidence: "\(article.decisionSummary.evidence) Source: \(article.article.sourceName).",
                confidence: article.decisionSummary.confidence,
                nextTest: nextTest(for: label),
                supportingArticle: article,
                createdAt: createdAt,
                isSavedToResearchQueue: false
            )
        }
    }

    private static func customer(for text: String, label: OpportunityLabel) -> String {
        if ["factory", "manufacturer", "industrial", "plc", "automation"].contains(where: text.contains) {
            return "Manufacturers and factory automation teams"
        }
        if ["developer", "api", "sdk", "open source"].contains(where: text.contains) {
            return "AI developers and technical product teams"
        }
        if ["enterprise", "procurement", "customer"].contains(where: text.contains) {
            return "Enterprise buyers and procurement teams"
        }
        return label == .marketMoving
            ? "Operators exposed to this market shift"
            : "Industrial AI product teams"
    }

    private static func trigger(for article: ScoredArticle, label: OpportunityLabel) -> String {
        let prefix = switch label {
        case .marketMoving: "Market-moving change"
        case .businessOpportunity: "Commercial trigger"
        case .aiLaunch: "Capability trigger"
        }
        return "\(prefix): \(article.decisionSummary.whatChanged)"
    }

    private static func wedge(for article: ScoredArticle, text: String) -> String {
        if [.robotics, .hardware, .tactile].contains(article.category)
            || ["factory", "plc", "automation"].contains(where: text.contains) {
            return "Pilot a narrow integration inside one measurable industrial workflow."
        }
        if article.category == .ai || ["api", "sdk", "developer"].contains(where: text.contains) {
            return "Adapt the capability into a focused developer workflow with a measurable baseline."
        }
        return "Validate a narrow buyer problem before committing product or research time."
    }

    private static func nextTest(for label: OpportunityLabel) -> String {
        switch label {
        case .marketMoving:
            "Interview three affected operators to test urgency and budget response."
        case .businessOpportunity:
            "Contact three likely buyers and validate the trigger, pain, and procurement timing."
        case .aiLaunch:
            "Prototype one narrow workflow and compare quality, latency, and cost with the current approach."
        }
    }
}
