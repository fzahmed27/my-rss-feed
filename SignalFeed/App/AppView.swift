import SwiftUI

struct AppView: View {
    var body: some View {
        TabView {
            NavigationStack {
                FeedView()
            }
            .tabItem {
                Label("Feed", systemImage: "newspaper")
            }

            NavigationStack {
                BriefingView()
            }
            .tabItem {
                Label("Briefing", systemImage: "sun.max")
            }

            NavigationStack {
                OpportunityInboxView()
            }
            .tabItem {
                Label("Opportunities", systemImage: "sparkles")
            }

            NavigationStack {
                SourcesView()
            }
            .tabItem {
                Label("Sources", systemImage: "antenna.radiowaves.left.and.right")
            }

            NavigationStack {
                PeopleView()
            }
            .tabItem {
                Label("People", systemImage: "person.2")
            }

            NavigationStack {
                SettingsView()
            }
            .tabItem {
                Label("Settings", systemImage: "slider.horizontal.3")
            }
        }
    }
}

#Preview {
    AppView()
        .environment(FeedStore.preview)
}
