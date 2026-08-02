import SwiftUI

struct SettingsView: View {
    @Environment(FeedStore.self) private var store
    @State private var feedbackExportURL: URL?
    @State private var exportError: String?

    var body: some View {
        Form {
            Section("Ranking") {
                Stepper(
                    value: Binding(
                        get: { store.presentationSettings.readingBudgetMinutes },
                        set: { store.updateReadingBudgetMinutes($0) }
                    ),
                    in: 5...60,
                    step: 5
                ) {
                    Label("\(store.presentationSettings.readingBudgetMinutes) minute reading budget", systemImage: "timer")
                }

                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Label("Minimum score", systemImage: "line.3.horizontal.decrease")
                        Spacer()
                        Text(store.settings.minimumScore.formatted(.number.precision(.fractionLength(1))))
                            .foregroundStyle(.secondary)
                    }

                    Slider(
                        value: Binding(
                            get: { store.settings.minimumScore },
                            set: { store.updateMinimumScore($0) }
                        ),
                        in: 0...30,
                        step: 0.5
                    )
                }

                Stepper(
                    value: Binding(
                        get: { store.settings.daysBack },
                        set: { store.updateDaysBack($0) }
                    ),
                    in: 1...90
                ) {
                    Label("\(store.settings.daysBack) day window", systemImage: "calendar")
                }

                Stepper(
                    value: Binding(
                        get: { store.settings.maxArticles },
                        set: { store.updateMaxArticles($0) }
                    ),
                    in: 20...200,
                    step: 10
                ) {
                    Label("\(store.settings.maxArticles) max items", systemImage: "number")
                }
            }

            Section("Extraction") {
                Toggle(
                    isOn: Binding(
                        get: { store.settings.isFullTextExtractionEnabled },
                        set: { store.updateFullTextExtractionEnabled($0) }
                    )
                ) {
                    Label("Full-text articles", systemImage: "doc.text.magnifyingglass")
                }

                Stepper(
                    value: Binding(
                        get: { store.settings.fullTextArticleLimit },
                        set: { store.updateFullTextArticleLimit($0) }
                    ),
                    in: 0...120,
                    step: 10
                ) {
                    Label("\(store.settings.fullTextArticleLimit) pages per refresh", systemImage: "doc.on.doc")
                }
                .disabled(!store.settings.isFullTextExtractionEnabled)
            }

            Section("Configuration") {
                NavigationLink {
                    FounderProfileView()
                } label: {
                    Label("Founder Profile", systemImage: "person.text.rectangle")
                }
                LabeledContent("Sources", value: "\(store.sources.count)")
                LabeledContent("Active sources", value: "\(store.activeSources.count)")
                LabeledContent("Muted sources", value: "\(store.sources.filter(\.isMuted).count)")
                LabeledContent("Weighted keywords", value: "\(DefaultContentConfig.keywords.count)")
                LabeledContent("Parser", value: "RSS + Atom")
                LabeledContent("Storage", value: "SwiftData")
            }

            Section("Learning Data") {
                LabeledContent("Saved articles", value: "\(store.bookmarkCount)")
                LabeledContent("Feedback votes", value: "\(store.feedbackCount)")
                LabeledContent("Article opens", value: "\(store.openedCount)")
                LabeledContent("Muted topics", value: "\(store.mutedTopics.count)")

                Button {
                    do {
                        feedbackExportURL = try store.makeFeedbackExportURL()
                        exportError = nil
                    } catch {
                        exportError = error.localizedDescription
                    }
                } label: {
                    Label("Prepare Feedback Export", systemImage: "square.and.arrow.up")
                }

                if let feedbackExportURL {
                    ShareLink(item: feedbackExportURL) {
                        Label("Share Feedback Export", systemImage: "doc.text")
                    }
                }

                if let exportError {
                    Text(exportError)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }

            Section {
                Button {
                    Task { await store.refresh() }
                } label: {
                    Label("Refresh with Current Settings", systemImage: "arrow.clockwise")
                }
                .disabled(store.isLoading)

                Button(role: .destructive) {
                    store.resetSettings()
                } label: {
                    Label("Restore Ranking Defaults", systemImage: "arrow.counterclockwise")
                }

                Button(role: .destructive) {
                    store.resetSources()
                    Task { await store.refresh() }
                } label: {
                    Label("Restore Default Sources", systemImage: "antenna.radiowaves.left.and.right")
                }
            }
        }
        .navigationTitle("Settings")
    }
}

private struct FounderProfileView: View {
    @Environment(FeedStore.self) private var store
    @State private var draft = FounderProfileDraft(profile: .industrialAIFounder)
    @State private var hasLoaded = false
    @State private var exportURL: URL?
    @State private var exportError: String?
    @State private var isConfirmingReset = false

    var body: some View {
        Form {
            Section {
                TextField("Company", text: $draft.company)
                    .textInputAutocapitalization(.words)
            } header: {
                Text("Company")
            } footer: {
                Text("Company and market language can increase personal relevance when it appears in an article.")
            }

            profileTermsSection(title: "Customers", prompt: "Manufacturers, factory operators", text: $draft.customers)
            profileTermsSection(title: "Product Areas", prompt: "Industrial automation, edge inference", text: $draft.productAreas)
            profileTermsSection(title: "Competitors", prompt: "Company A, Company B", text: $draft.competitors)
            profileTermsSection(title: "Current Priorities", prompt: "Commercialization, predictive maintenance", text: $draft.priorities)
            profileTermsSection(title: "Topics To Track", prompt: "PLCs, PID tuning, sensors", text: $draft.topics)

            Section("Profile Data") {
                Button { prepareExport() } label: {
                    Label("Prepare Profile Export", systemImage: "square.and.arrow.up")
                }
                if let exportURL {
                    ShareLink(item: exportURL) {
                        Label("Share Profile Export", systemImage: "doc.text")
                    }
                }
                if let exportError {
                    Text(exportError).font(.caption).foregroundStyle(.red)
                }
                Button(role: .destructive) { isConfirmingReset = true } label: {
                    Label("Reset Founder Profile", systemImage: "arrow.counterclockwise")
                }
            }
        }
        .navigationTitle("Founder Profile")
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Save") { store.updateFounderProfile(draft.profile) }
                    .fontWeight(.semibold)
            }
        }
        .task {
            guard !hasLoaded else { return }
            draft = FounderProfileDraft(profile: store.founderProfile)
            hasLoaded = true
        }
        .confirmationDialog("Reset founder profile?", isPresented: $isConfirmingReset, titleVisibility: .visible) {
            Button("Reset to Industrial AI Defaults", role: .destructive) {
                store.resetFounderProfile()
                draft = FounderProfileDraft(profile: store.founderProfile)
                exportURL = nil
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This replaces company context and tracked terms with the default industrial AI founder profile.")
        }
    }

    private func profileTermsSection(title: String, prompt: String, text: Binding<String>) -> some View {
        Section {
            TextField(prompt, text: text, axis: .vertical)
                .lineLimit(2...4)
                .textInputAutocapitalization(.never)
        } header: {
            Text(title)
        } footer: {
            Text("Separate terms with commas.")
        }
    }

    private func prepareExport() {
        store.updateFounderProfile(draft.profile)
        do {
            exportURL = try store.makeFounderProfileExportURL()
            exportError = nil
        } catch {
            exportError = error.localizedDescription
        }
    }
}

private struct FounderProfileDraft {
    var company: String
    var customers: String
    var productAreas: String
    var competitors: String
    var priorities: String
    var topics: String

    init(profile: FounderContextProfile) {
        company = profile.company
        customers = profile.customers.joined(separator: ", ")
        productAreas = profile.productAreas.joined(separator: ", ")
        competitors = profile.competitors.joined(separator: ", ")
        priorities = profile.priorities.joined(separator: ", ")
        topics = profile.topics.joined(separator: ", ")
    }

    var profile: FounderContextProfile {
        FounderContextProfile(
            company: company.trimmingCharacters(in: .whitespacesAndNewlines),
            customers: terms(from: customers),
            productAreas: terms(from: productAreas),
            competitors: terms(from: competitors),
            priorities: terms(from: priorities),
            topics: terms(from: topics)
        )
    }

    private func terms(from text: String) -> [String] {
        var seen = Set<String>()
        return text.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && seen.insert($0.lowercased()).inserted }
    }
}

#Preview {
    NavigationStack {
        SettingsView()
            .environment(FeedStore.preview)
    }
}
