import SwiftUI

struct PeopleView: View {
    @Environment(FeedStore.self) private var store

    var body: some View {
        List {
            let people = store.sources.filter { $0.kind == .person }.sorted { $0.name < $1.name }

            if people.isEmpty {
                ContentUnavailableView(
                    "No people yet",
                    systemImage: "person.2",
                    description: Text("Add a source and set its type to Person.")
                )
            } else {
                Section("People") {
                    ForEach(people) { source in
                        NavigationLink {
                            PersonDetailView(source: source)
                        } label: {
                            PersonRow(source: source, articleCount: articleCount(for: source))
                        }
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("People")
        .task {
            await store.refreshIfNeeded()
        }
    }

    private func articleCount(for source: FeedSource) -> Int {
        store.result?.articles.filter { $0.article.sourceID == source.id }.count ?? 0
    }
}

private struct PersonRow: View {
    let source: FeedSource
    let articleCount: Int

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "person.crop.circle")
                .font(.title3)
                .foregroundStyle(.white)
                .frame(width: 38, height: 38)
                .background(Color.accentColor, in: RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 4) {
                Text(source.name)
                    .font(.headline)
                Text(source.url)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer()

            Text("\(articleCount)")
                .font(.subheadline.weight(.bold))
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }
}

private struct PersonDetailView: View {
    @Environment(FeedStore.self) private var store
    let source: FeedSource
    @State private var safariDestination: SafariDestination?
    @State private var selectedArticle: ScoredArticle?

    var body: some View {
        let articles = store.result?.articles.filter { $0.article.sourceID == source.id } ?? []

        List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Label(source.name, systemImage: source.kind.systemImage)
                        .font(.headline)
                    Text(source.url)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    LabeledContent("Reputation", value: source.reputation.formatted(.number.precision(.fractionLength(2))))
                }
                .padding(.vertical, 4)
            }

            if articles.isEmpty {
                ContentUnavailableView(
                    "No ranked items",
                    systemImage: "person.text.rectangle",
                    description: Text("Refresh the feed or lower the minimum score to see this person's items.")
                )
            } else {
                Section("Ranked Items") {
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
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle(source.name)
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
        PeopleView()
            .environment(FeedStore.preview)
    }
}
