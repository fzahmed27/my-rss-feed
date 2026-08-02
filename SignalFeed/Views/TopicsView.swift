import SwiftUI

struct TopicsView: View {
    @Environment(FeedStore.self) private var store

    var body: some View {
        content
            .navigationTitle("Topics")
            .task {
                await store.refreshIfNeeded()
            }
    }

    @ViewBuilder
    private var content: some View {
        if let result = store.result {
            List {
                ForEach(topicRows(from: result), id: \.category) { row in
                    NavigationLink {
                        TopicDetailView(category: row.category, articles: row.articles)
                    } label: {
                        TopicRow(category: row.category, articles: row.articles)
                    }
                }
            }
            .listStyle(.insetGrouped)
        } else {
            ContentUnavailableView(
                "No topics yet",
                systemImage: "square.grid.2x2",
                description: Text("Refresh the feed to group ranked items by topic.")
            )
        }
    }

    private func topicRows(from result: DigestResult) -> [(category: ArticleCategory, articles: [ScoredArticle])] {
        ArticleCategory.allCases.compactMap { category in
            guard let articles = result.articlesByCategory[category], !articles.isEmpty else {
                return nil
            }
            return (category, articles)
        }
    }
}

private struct TopicRow: View {
    let category: ArticleCategory
    let articles: [ScoredArticle]

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: category.systemImage)
                .font(.title3)
                .foregroundStyle(.white)
                .frame(width: 38, height: 38)
                .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 4) {
                Text(category.title)
                    .font(.headline)
                Text(topTitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            Text("\(articles.count)")
                .font(.subheadline.weight(.bold))
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }

    private var topTitle: String {
        articles.first?.article.title ?? "No items"
    }
}

private struct TopicDetailView: View {
    @Environment(FeedStore.self) private var store
    let category: ArticleCategory
    let articles: [ScoredArticle]
    @State private var safariDestination: SafariDestination?
    @State private var selectedArticle: ScoredArticle?

    var body: some View {
        List {
            ForEach(articles) { article in
                ArticleCard(
                    article: article,
                    isBookmarked: store.isBookmarked(article),
                    interaction: store.interaction(for: article),
                    onSelect: {
                        selectedArticle = article
                    },
                    onOpen: {
                        guard let url = URL(string: article.article.link) else { return }
                        store.recordOpen(for: article)
                        safariDestination = SafariDestination(url: url)
                    },
                    onToggleBookmark: {
                        store.toggleBookmark(for: article)
                    },
                    onFeedback: { feedback, reasons in
                        store.setFeedback(feedback, reasons: reasons, for: article)
                    },
                    onClearFeedback: {
                        store.clearFeedback(for: article)
                    },
                    onToggleRead: {
                        store.markRead(article, isRead: !store.isRead(article))
                    },
                    onDismiss: {
                        store.dismiss(article)
                    },
                    onMuteTopic: {
                        store.mutePrimaryTopic(for: article)
                        Task { await store.refresh() }
                    }
                )
                .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                .listRowSeparator(.hidden)
            }
        }
        .listStyle(.plain)
        .navigationTitle(category.title)
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $safariDestination) { destination in
            SafariView(url: destination.url)
        }
        .sheet(item: $selectedArticle) { article in
            ArticleDetailView(article: article)
        }
    }
}

#Preview {
    NavigationStack {
        TopicsView()
            .environment(FeedStore.preview)
    }
}
