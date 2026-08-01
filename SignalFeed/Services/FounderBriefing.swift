import Foundation

enum FounderBriefingSlot: String, CaseIterable, Codable, Identifiable, Sendable {
    case aiCapability
    case industrialAutomation
    case developerTool
    case startupSignal
    case scienceOrSpace

    var id: String { rawValue }

    var title: String {
        switch self {
        case .aiCapability: "AI capability change"
        case .industrialAutomation: "Industrial automation"
        case .developerTool: "Developer-tool update"
        case .startupSignal: "Startup or commercialization"
        case .scienceOrSpace: "Science or space"
        }
    }

    var subtitle: String {
        switch self {
        case .aiCapability: "What changed in model, agent, inference, or applied AI capability."
        case .industrialAutomation: "Controls, PLCs, robotics, sensors, factories, or predictive maintenance."
        case .developerTool: "Tools that could change how you build or ship."
        case .startupSignal: "Commercial traction, customer, funding, procurement, or GTM signal. Not financial advice."
        case .scienceOrSpace: "High-value science or space context without forcing a weak item."
        }
    }

    var systemImage: String {
        switch self {
        case .aiCapability: "sparkles"
        case .industrialAutomation: "gearshape.2"
        case .developerTool: "hammer"
        case .startupSignal: "building.2"
        case .scienceOrSpace: "globe.americas"
        }
    }

    func matches(_ article: ScoredArticle) -> Bool {
        let text = article.article.searchableText
        switch self {
        case .aiCapability:
            return text.contains("model") || text.contains("agent") || text.contains("inference") || text.contains("benchmark") || article.category == .ai
        case .industrialAutomation:
            return ["industrial", "automation", "plc", "pid", "control system", "factory", "sensor", "predictive maintenance"].contains { text.contains($0) }
                || [.robotics, .hardware, .tactile].contains(article.category)
        case .developerTool:
            return ["developer tools", "api", "sdk", "open source", "pytorch", "tooling"].contains { text.contains($0) }
        case .startupSignal:
            return article.category == .startups || ["customer", "commercialization", "go-to-market", "funding", "procurement", "enterprise adoption"].contains { text.contains($0) }
        case .scienceOrSpace:
            return ["space", "science", "nasa", "satellite", "physics", "materials"].contains { text.contains($0) }
        }
    }
}

struct FounderBriefingItem: Identifiable, Codable, Hashable, Sendable {
    let rank: Int
    let slot: FounderBriefingSlot
    let article: ScoredArticle?
    let emptyReason: String

    var id: String { slot.rawValue }
}

struct FounderBriefingSnapshot: Identifiable, Codable, Hashable, Sendable {
    let id: UUID
    let generatedAt: Date
    let savedAt: Date
    let intent: IntentMode
    let readingBudgetMinutes: Int
    let fetchedCount: Int
    let items: [FounderBriefingItem]
    var lastSharedAt: Date?
    var shareCount: Int

    init(
        id: UUID = UUID(),
        generatedAt: Date,
        savedAt: Date = Date(),
        intent: IntentMode,
        readingBudgetMinutes: Int,
        fetchedCount: Int,
        items: [FounderBriefingItem],
        lastSharedAt: Date? = nil,
        shareCount: Int = 0
    ) {
        self.id = id
        self.generatedAt = generatedAt
        self.savedAt = savedAt
        self.intent = intent
        self.readingBudgetMinutes = readingBudgetMinutes
        self.fetchedCount = fetchedCount
        self.items = items
        self.lastSharedAt = lastSharedAt
        self.shareCount = shareCount
    }

    var exportText: String {
        let metadata = """
        Signal Feed Founder Briefing
        Generated: \(ISO8601DateFormatter().string(from: generatedAt))
        Intent: \(intent.title)
        Reading budget: \(readingBudgetMinutes) minutes
        Fetched items: \(fetchedCount)
        """
        return metadata + "\n\n" + FounderBriefingGenerator.exportText(items: items)
    }

    func hasSameContent(
        generatedAt: Date,
        intent: IntentMode,
        readingBudgetMinutes: Int,
        items: [FounderBriefingItem]
    ) -> Bool {
        self.generatedAt == generatedAt
            && self.intent == intent
            && self.readingBudgetMinutes == readingBudgetMinutes
            && self.items == items
    }
}

struct BriefingCompletion: Identifiable, Codable, Hashable, Sendable {
    let id: UUID
    let snapshotID: UUID
    let generatedAt: Date
    let completedAt: Date
    let intent: IntentMode
    let readingBudgetMinutes: Int
    let reviewedSlotCount: Int

    init(
        id: UUID = UUID(),
        snapshotID: UUID,
        generatedAt: Date,
        completedAt: Date = Date(),
        intent: IntentMode,
        readingBudgetMinutes: Int,
        reviewedSlotCount: Int
    ) {
        self.id = id
        self.snapshotID = snapshotID
        self.generatedAt = generatedAt
        self.completedAt = completedAt
        self.intent = intent
        self.readingBudgetMinutes = readingBudgetMinutes
        self.reviewedSlotCount = reviewedSlotCount
    }
}

enum FounderBriefingGenerator {
    static func generate(from articles: [ScoredArticle]) -> [FounderBriefingItem] {
        var usedIDs = Set<String>()
        return FounderBriefingSlot.allCases.enumerated().map { index, slot in
            let candidate = articles
                .filter { !usedIDs.contains($0.id) && slot.matches($0) }
                .filter { $0.score >= 45 || !$0.opportunityLabels.isEmpty }
                .sorted { $0.score > $1.score }
                .first

            if let candidate {
                usedIDs.insert(candidate.id)
            }

            return FounderBriefingItem(
                rank: index + 1,
                slot: slot,
                article: candidate,
                emptyReason: "No recent source cleared the quality bar for this category."
            )
        }
    }

    static func exportText(items: [FounderBriefingItem]) -> String {
        items.map { item in
            guard let article = item.article else {
                return "\(item.rank). \(item.slot.title): No strong item."
            }
            return """
            \(item.rank). \(item.slot.title): \(article.article.title)
            What happened: \(article.decisionSummary.whatChanged)
            Why it matters: \(article.decisionSummary.whyItMattersToYou)
            Should I care?: \(article.decisionSummary.shouldCare)
            What you should do: \(article.decisionSummary.suggestedAction)
            Confidence: \(article.decisionSummary.confidence.title)
            Evidence: \(article.decisionSummary.evidence)
            """
        }
        .joined(separator: "\n\n")
    }
}
