import SwiftUI

struct ArticleDetailView: View {
    @Environment(FeedStore.self) private var store
    @Environment(\.dismiss) private var dismiss
    let article: ScoredArticle
    @State private var safariDestination: SafariDestination?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    header
                    actionBar
                    summarySection
                    reasonSection
                    sourceCoverageSection
                    metadataSection
                }
                .padding(16)
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Why This Ranked")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .sheet(item: $safariDestination) { destination in
                SafariView(url: destination.url)
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 8) {
                    Text(article.article.title)
                        .font(.title3.weight(.bold))
                        .fixedSize(horizontal: false, vertical: true)

                    HStack(spacing: 8) {
                        Label(article.article.sourceName, systemImage: article.article.sourceKind.systemImage)
                        if let published = article.article.publishedAt {
                            Label(published.formatted(.relative(presentation: .named)), systemImage: "clock")
                        }
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }

                Spacer(minLength: 8)

                VStack(spacing: 2) {
                    Text(article.score.formatted(.number.precision(.fractionLength(1))))
                        .font(.headline.weight(.bold))
                    Text("score")
                        .font(.caption2.weight(.medium))
                }
                .foregroundStyle(.white)
                .frame(width: 58, height: 48)
                .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 8))
            }

            HStack(spacing: 8) {
                Label(article.category.title, systemImage: article.category.systemImage)
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))

                ForEach(article.opportunityLabels) { label in
                    Label(label.title, systemImage: label.systemImage)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .foregroundStyle(.white)
                        .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
        .padding(14)
        .background(.background, in: RoundedRectangle(cornerRadius: 8))
    }

    private var actionBar: some View {
        HStack(spacing: 10) {
            Button {
                store.toggleBookmark(for: article)
            } label: {
                Label(store.isBookmarked(article) ? "Saved" : "Save", systemImage: store.isBookmarked(article) ? "bookmark.fill" : "bookmark")
                    .frame(maxWidth: .infinity)
            }

            Button {
                store.setFeedback(.liked, for: article)
            } label: {
                Label("Useful", systemImage: store.feedback(for: article) == .liked ? "hand.thumbsup.fill" : "hand.thumbsup")
                    .frame(maxWidth: .infinity)
            }

            Button {
                store.setFeedback(.disliked, for: article)
            } label: {
                Label("No", systemImage: store.feedback(for: article) == .disliked ? "hand.thumbsdown.fill" : "hand.thumbsdown")
                    .frame(maxWidth: .infinity)
            }
        }
        .font(.subheadline.weight(.semibold))
        .buttonStyle(.bordered)
        .controlSize(.regular)
    }

    @ViewBuilder
    private var summarySection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Signal Contract")
                .font(.headline)

            DetailDecisionLine(title: "What happened", value: article.decisionSummary.whatChanged)
            DetailDecisionLine(title: "Why it matters", value: article.decisionSummary.whyItMattersToYou)
            DetailDecisionLine(title: "Should I care?", value: article.decisionSummary.shouldCare)
            DetailDecisionLine(title: "Suggested action", value: article.decisionSummary.suggestedAction)
            DetailDecisionLine(title: "Confidence", value: "\(article.decisionSummary.confidence.title). \(article.decisionSummary.primaryUncertainty)")
            DetailDecisionLine(title: "Evidence", value: article.decisionSummary.evidence)

            if !article.article.summary.isEmpty {
                Divider()
                Text(article.article.summary)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.background, in: RoundedRectangle(cornerRadius: 8))
    }

    private var reasonSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Ranking Reasons")
                .font(.headline)

            ForEach(article.reasonDetails) { reason in
                ReasonDetailRow(reason: reason)
            }

            if !article.opportunityLabels.isEmpty {
                Text("Opportunity flags are screening signals only, not financial advice.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 2)
            }
        }
        .padding(14)
        .background(.background, in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private var sourceCoverageSection: some View {
        if let cluster = article.eventCluster {
            VStack(alignment: .leading, spacing: 12) {
                Label("\(cluster.sourceCount) Sources", systemImage: "doc.on.doc")
                    .font(.headline)
                Text("Signal Feed selected \(article.article.sourceName) as the representative source.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                ForEach(cluster.alternateArticles) { alternate in
                    Button {
                        guard let url = URL(string: alternate.link) else { return }
                        safariDestination = SafariDestination(url: url)
                    } label: {
                        HStack(alignment: .center, spacing: 10) {
                            Image(systemName: alternate.sourceKind.systemImage)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(alternate.sourceName)
                                    .font(.subheadline.weight(.semibold))
                                Text(alternate.title)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                            Spacer()
                            Image(systemName: "arrow.up.right")
                                .foregroundStyle(.secondary)
                        }
                        .frame(minHeight: 44)
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Open alternate coverage from \(alternate.sourceName)")
                }
            }
            .padding(14)
            .background(.background, in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var metadataSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Source & Signals")
                .font(.headline)

            LabeledContent("Source trust", value: article.article.sourceReputation.formatted(.number.precision(.fractionLength(2))))
            LabeledContent("Full text", value: article.article.hasExtractedContent ? "\(article.article.extractedContent.count.formatted()) chars" : "RSS only")
            LabeledContent("Canonical URL", value: article.canonicalURL)
            LabeledContent("Matched keywords", value: article.matchedKeywords.isEmpty ? "None" : article.matchedKeywords.joined(separator: ", "))

            Button {
                openArticle()
            } label: {
                Label("Open Article", systemImage: "safari")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 4)
        }
        .font(.subheadline)
        .padding(14)
        .background(.background, in: RoundedRectangle(cornerRadius: 8))
    }

    private func openArticle() {
        guard let url = URL(string: article.article.link) else { return }
        store.recordOpen(for: article)
        safariDestination = SafariDestination(url: url)
    }
}

private struct DetailDecisionLine: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline)
                .foregroundStyle(.primary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct ReasonDetailRow: View {
    let reason: RankingReason

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: reason.kind.systemImage)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
                .frame(width: 30, height: 30)
                .background(color, in: RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 4) {
                HStack(alignment: .firstTextBaseline) {
                    Text(reason.title)
                        .font(.subheadline.weight(.semibold))
                    Spacer()
                    Text(formattedImpact)
                        .font(.caption.weight(.bold))
                        .foregroundStyle(impactColor)
                }

                Text(reason.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var color: Color {
        switch reason.kind {
        case .sourceTrust: .green
        case .keyword: .blue
        case .fullText: .indigo
        case .freshness: .orange
        case .opportunity: .purple
        case .duplicate: .gray
        case .personal: .teal
        case .baseline: .secondary
        }
    }

    private var formattedImpact: String {
        if reason.impact == 0 {
            return "0"
        }

        let value = reason.impact.formatted(.number.precision(.fractionLength(1)))
        return reason.impact > 0 ? "+\(value)" : value
    }

    private var impactColor: Color {
        if reason.impact < 0 {
            return .red
        }

        return reason.impact == 0 ? .secondary : .accentColor
    }
}

#Preview {
    ArticleDetailView(article: .placeholder)
        .environment(FeedStore.preview)
}
