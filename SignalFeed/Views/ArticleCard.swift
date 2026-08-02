import SwiftUI

struct ArticleCard: View {
    let article: ScoredArticle
    let isBookmarked: Bool
    let interaction: ArticleInteraction
    let onSelect: () -> Void
    let onOpen: () -> Void
    let onToggleBookmark: () -> Void
    let onFeedback: (ArticleFeedback, [FeedbackReason]) -> Void
    let onClearFeedback: () -> Void
    let onToggleRead: () -> Void
    let onDismiss: () -> Void
    let onMuteTopic: () -> Void

    @State private var feedbackChoice: ArticleFeedback?
    @State private var selectedReasons: Set<FeedbackReason> = []
    @State private var showsRankingDetails = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .top, spacing: 8) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(article.article.title)
                            .font(.headline)
                            .foregroundStyle(.primary)
                            .lineLimit(3)

                        HStack(spacing: 8) {
                            Label(article.article.sourceName, systemImage: article.article.sourceKind.systemImage)
                            if let published = article.article.publishedAt {
                                Label(published.formatted(.relative(presentation: .named)), systemImage: "clock")
                            }
                            Label("\(article.decisionSummary.estimatedReadingMinutes) min read", systemImage: "timer")
                        }
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    }

                    Spacer(minLength: 8)

                    RelevanceBadge(label: article.relevanceLabel, score: article.score)
                }

                if !article.article.summary.isEmpty {
                    Text(article.article.summary)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(3)
                }

                HStack(spacing: 8) {
                    Label(article.category.title, systemImage: article.category.systemImage)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))

                    ForEach(article.opportunityLabels.prefix(2)) { label in
                        Label(label.title, systemImage: label.systemImage)
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 5)
                            .foregroundStyle(.white)
                            .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 8))
                    }
                }
                .lineLimit(1)

                if let cluster = article.eventCluster {
                    DisclosureGroup {
                        VStack(alignment: .leading, spacing: 8) {
                            ForEach(cluster.alternateArticles) { alternate in
                                if let url = URL(string: alternate.link) {
                                    Link(destination: url) {
                                        HStack(alignment: .firstTextBaseline) {
                                            Label(alternate.sourceName, systemImage: alternate.sourceKind.systemImage)
                                            Spacer()
                                            Text(alternate.sourceReputation.formatted(.number.precision(.fractionLength(2))))
                                                .monospacedDigit()
                                        }
                                        .font(.caption)
                                        .frame(minHeight: 44)
                                    }
                                    .accessibilityLabel("Open alternate coverage from \(alternate.sourceName)")
                                }
                            }
                        }
                        .padding(.top, 4)
                    } label: {
                        Label("\(cluster.sourceCount) sources covering this event", systemImage: "doc.on.doc")
                            .font(.caption.weight(.semibold))
                    }
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text(article.decisionSummary.whyThisMatters)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)
                        .lineLimit(2)

                    DecisionLine(title: "What changed", value: article.decisionSummary.whatChanged)
                    DecisionLine(title: "Why it matters to you", value: article.decisionSummary.whyItMattersToYou)
                    DecisionLine(title: "Should I care?", value: article.decisionSummary.shouldCare)
                    DecisionLine(title: "Suggested action", value: article.decisionSummary.suggestedAction)

                    HStack(spacing: 8) {
                        Label(article.decisionSummary.confidence.title, systemImage: "checkmark.shield")
                        Text(article.decisionSummary.evidence)
                    }
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                }

                DisclosureGroup(isExpanded: $showsRankingDetails) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(article.decisionSummary.primaryUncertainty)
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        ForEach(article.scoreComponents) { component in
                            HStack(alignment: .top) {
                                Text(component.kind.title)
                                Spacer()
                                Text(component.contribution.formatted(.number.precision(.fractionLength(1))))
                                    .monospacedDigit()
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }

                        ForEach(article.penalties.filter { $0.value > 0 }) { penalty in
                            HStack(alignment: .top) {
                                Text(penalty.title)
                                Spacer()
                                Text("-\(penalty.value.formatted(.number.precision(.fractionLength(1))))")
                                    .monospacedDigit()
                            }
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.top, 4)
                } label: {
                    Label("Ranking details", systemImage: "slider.horizontal.3")
                        .font(.caption.weight(.semibold))
                }
            }
            .contentShape(Rectangle())
            .onTapGesture(perform: onSelect)

            Divider()

            HStack(spacing: 10) {
                Button(action: onToggleBookmark) {
                    Image(systemName: isBookmarked ? "bookmark.fill" : "bookmark")
                        .frame(width: 34, height: 30)
                }
                .accessibilityLabel(isBookmarked ? "Remove bookmark" : "Bookmark article")

                Button {
                    showFeedback(.liked)
                } label: {
                    Image(systemName: interaction.feedback == .liked ? "hand.thumbsup.fill" : "hand.thumbsup")
                        .frame(width: 44, height: 36)
                }
                .accessibilityLabel("Mark useful")

                Button {
                    showFeedback(.disliked)
                } label: {
                    Image(systemName: interaction.feedback == .disliked ? "hand.thumbsdown.fill" : "hand.thumbsdown")
                        .frame(width: 44, height: 36)
                }
                .accessibilityLabel("Mark not useful")

                Button(action: onToggleRead) {
                    Image(systemName: interaction.isRead ? "circle.fill" : "circle")
                        .frame(width: 44, height: 36)
                }
                .accessibilityLabel(interaction.isRead ? "Mark unread" : "Mark read")

                Menu {
                    Button(action: onDismiss) {
                        Label("Dismiss", systemImage: "xmark")
                    }
                    Button(action: onMuteTopic) {
                        Label("Mute topic for 30 days", systemImage: "speaker.slash")
                    }
                    if interaction.feedback != nil {
                        Button(action: onClearFeedback) {
                            Label("Undo feedback", systemImage: "arrow.uturn.backward")
                        }
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                        .frame(width: 44, height: 36)
                }
                .accessibilityLabel("More article actions")

                Spacer()

                Button(action: onOpen) {
                    Image(systemName: "safari")
                        .frame(width: 34, height: 30)
                }
                .accessibilityLabel("Open article")
            }
            .font(.subheadline.weight(.semibold))
            .foregroundStyle(.primary)
            .buttonStyle(.borderless)
            .tint(.accentColor)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(.quaternary, lineWidth: 1)
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel("\(article.article.title), \(article.relevanceLabel), estimated \(article.decisionSummary.estimatedReadingMinutes) minute read")
        .sheet(item: $feedbackChoice) { feedback in
            FeedbackReasonSheet(
                feedback: feedback,
                selectedReasons: $selectedReasons,
                onCancel: {
                    feedbackChoice = nil
                },
                onSave: {
                    onFeedback(feedback, Array(selectedReasons))
                    feedbackChoice = nil
                }
            )
        }
    }

    private func showFeedback(_ feedback: ArticleFeedback) {
        feedbackChoice = feedback
        selectedReasons = Set(interaction.feedback == feedback ? interaction.feedbackReasons : [])
    }
}

private struct DecisionLine: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
    }
}

private struct RelevanceBadge: View {
    let label: String
    let score: Double

    var body: some View {
        VStack(spacing: 2) {
            Text(label.replacingOccurrences(of: " Relevance", with: ""))
                .font(.caption.weight(.bold))
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text("\(articleScore)")
                .font(.caption2.weight(.medium))
        }
        .foregroundStyle(.white)
        .frame(width: 76, height: 44)
        .background(scoreColor, in: RoundedRectangle(cornerRadius: 8))
    }

    private var articleScore: String {
        score.formatted(.number.precision(.fractionLength(0)))
    }

    private var scoreColor: Color {
        switch score {
        case 72...:
            .green
        case 45..<72:
            .blue
        default:
            .secondary
        }
    }
}

private struct FeedbackReasonSheet: View {
    let feedback: ArticleFeedback
    @Binding var selectedReasons: Set<FeedbackReason>
    let onCancel: () -> Void
    let onSave: () -> Void

    private var reasons: [FeedbackReason] {
        switch feedback {
        case .liked:
            [.relevantToCurrentProject, .strongBusinessSignal, .deepTechnicalInsight, .trackThisTopic, .trackThisCompanyOrPerson]
        case .disliked:
            [.tooGeneric, .notRelevantToMyCompany, .tooAcademic, .alreadyKnewThis, .lowQualitySource, .interestingButNotNow]
        }
    }

    var body: some View {
        NavigationStack {
            List {
                Section(feedback.title) {
                    ForEach(reasons) { reason in
                        Button {
                            if selectedReasons.contains(reason) {
                                selectedReasons.remove(reason)
                            } else {
                                selectedReasons.insert(reason)
                            }
                        } label: {
                            Label(
                                reason.title,
                                systemImage: selectedReasons.contains(reason) ? "checkmark.circle.fill" : "circle"
                            )
                        }
                    }
                }
            }
            .navigationTitle("Feedback")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel", action: onCancel)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save", action: onSave)
                }
            }
        }
        .presentationDetents([.medium])
    }
}

#Preview("Long article title") {
    ArticleCard(
        article: FounderModePreviewFixtures.longTitleArticle,
        isBookmarked: true,
        interaction: ArticleInteraction(
            isBookmarked: true,
            feedback: .liked,
            feedbackReasons: [.deepTechnicalInsight],
            isRead: false,
            isDismissed: false,
            openedCount: 0,
            updatedAt: Date()
        ),
        onSelect: {},
        onOpen: {},
        onToggleBookmark: {},
        onFeedback: { _, _ in },
        onClearFeedback: {},
        onToggleRead: {},
        onDismiss: {},
        onMuteTopic: {}
    )
        .padding()
}

#Preview("Accessibility text") {
    ScrollView {
        ArticleCard(
            article: FounderModePreviewFixtures.longTitleArticle,
            isBookmarked: false,
            interaction: .empty,
            onSelect: {},
            onOpen: {},
            onToggleBookmark: {},
            onFeedback: { _, _ in },
            onClearFeedback: {},
            onToggleRead: {},
            onDismiss: {},
            onMuteTopic: {}
        )
        .padding()
    }
    .environment(\.dynamicTypeSize, .accessibility5)
}
