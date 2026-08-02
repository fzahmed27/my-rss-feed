import SwiftUI

@main
struct SignalFeedApp: App {
    @State private var store = FeedStore()

    var body: some Scene {
        WindowGroup {
            AppView()
                .environment(store)
        }
    }
}
