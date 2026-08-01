import Foundation

struct DiscoveredFeed: Identifiable, Hashable, Sendable {
    let title: String
    let url: String
    let websiteURL: String

    var id: String { url }
}

enum FeedDiscoveryService {
    static func discover(from rawValue: String) async -> [DiscoveredFeed] {
        guard let websiteURL = normalizedWebsiteURL(from: rawValue) else { return [] }

        var candidates: [DiscoveredFeed] = [
            DiscoveredFeed(
                title: defaultTitle(for: websiteURL),
                url: normalizedAbsoluteString(websiteURL),
                websiteURL: normalizedAbsoluteString(websiteURL)
            )
        ]

        if let html = await fetchHTML(from: websiteURL) {
            let pageTitle = pageTitle(from: html) ?? defaultTitle(for: websiteURL)
            candidates.append(contentsOf: feedsFromHTML(html, baseURL: websiteURL, pageTitle: pageTitle))
        }

        candidates.append(contentsOf: commonFeedCandidates(for: websiteURL))
        let uniqueCandidates = unique(candidates)

        return await withTaskGroup(of: DiscoveredFeed?.self) { group in
            for candidate in uniqueCandidates {
                group.addTask {
                    await isValidFeed(candidate.url) ? candidate : nil
                }
            }

            var discovered: [DiscoveredFeed] = []
            for await candidate in group {
                if let candidate {
                    discovered.append(candidate)
                }
            }

            return unique(discovered).sorted { $0.title < $1.title }
        }
    }

    private static func fetchHTML(from url: URL) async -> String? {
        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 10
            request.setValue("SignalFeed-iOS/1.0", forHTTPHeaderField: "User-Agent")
            request.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")

            let (data, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse,
               !(200..<300).contains(httpResponse.statusCode) {
                return nil
            }

            let capped = data.prefix(1_200_000)
            return String(data: capped, encoding: .utf8)
                ?? String(data: capped, encoding: .isoLatin1)
        } catch {
            return nil
        }
    }

    private static func isValidFeed(_ rawURL: String) async -> Bool {
        guard let url = URL(string: rawURL),
              ["http", "https"].contains(url.scheme?.lowercased() ?? "") else {
            return false
        }

        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 8
            request.setValue("SignalFeed-iOS/1.0", forHTTPHeaderField: "User-Agent")
            request.setValue("application/rss+xml,application/atom+xml,application/xml,text/xml,*/*", forHTTPHeaderField: "Accept")

            let (data, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse,
               !(200..<300).contains(httpResponse.statusCode) {
                return false
            }

            let capped = data.prefix(80_000)
            let body = (String(data: capped, encoding: .utf8)
                ?? String(data: capped, encoding: .isoLatin1)
                ?? "")
                .lowercased()

            return body.contains("<rss")
                || body.contains("<feed")
                || body.contains("<rdf:rdf")
        } catch {
            return false
        }
    }

    private static func feedsFromHTML(
        _ html: String,
        baseURL: URL,
        pageTitle: String
    ) -> [DiscoveredFeed] {
        linkTags(in: html).compactMap { tag in
            let attrs = attributes(in: tag)
            let rel = attrs["rel", default: ""].lowercased()
            let type = attrs["type", default: ""].lowercased()
            guard let href = attrs["href"], !href.isEmpty else { return nil }

            let looksLikeFeed = rel.contains("alternate")
                && (type.contains("rss")
                    || type.contains("atom")
                    || type.contains("xml")
                    || href.lowercased().contains("feed")
                    || href.lowercased().hasSuffix(".xml"))
            guard looksLikeFeed,
                  let resolved = URL(string: htmlDecoded(href), relativeTo: baseURL)?.absoluteURL else {
                return nil
            }

            let title = attrs["title"].map { htmlDecoded($0).collapsedWhitespace() }
            return DiscoveredFeed(
                title: title?.isEmpty == false ? title! : pageTitle,
                url: normalizedAbsoluteString(resolved),
                websiteURL: normalizedAbsoluteString(baseURL)
            )
        }
    }

    private static func commonFeedCandidates(for url: URL) -> [DiscoveredFeed] {
        let root = rootURL(for: url)
        let title = defaultTitle(for: root)
        return ["/feed", "/feed.xml", "/rss.xml", "/atom.xml", "/index.xml"].compactMap { path in
            guard let feedURL = URL(string: path, relativeTo: root)?.absoluteURL else { return nil }
            return DiscoveredFeed(
                title: title,
                url: normalizedAbsoluteString(feedURL),
                websiteURL: normalizedAbsoluteString(root)
            )
        }
    }

    private static func rootURL(for url: URL) -> URL {
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: true) else {
            return url
        }

        components.path = "/"
        components.query = nil
        components.fragment = nil
        return components.url ?? url
    }

    private static func normalizedWebsiteURL(from rawValue: String) -> URL? {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let candidate = trimmed.contains("://") ? trimmed : "https://\(trimmed)"
        guard let url = URL(string: candidate),
              ["http", "https"].contains(url.scheme?.lowercased() ?? "") else {
            return nil
        }

        return url
    }

    private static func normalizedAbsoluteString(_ url: URL) -> String {
        guard var components = URLComponents(url: url, resolvingAgainstBaseURL: true) else {
            return url.absoluteString
        }
        components.fragment = nil
        return components.url?.absoluteString ?? url.absoluteString
    }

    private static func pageTitle(from html: String) -> String? {
        guard let title = firstMatch(in: html, pattern: "(?is)<title[^>]*>(.*?)</title>") else {
            return nil
        }
        let cleaned = title.removingHTML().collapsedWhitespace()
        return cleaned.isEmpty ? nil : cleaned
    }

    private static func defaultTitle(for url: URL) -> String {
        url.host?
            .replacingOccurrences(of: "www.", with: "")
            .components(separatedBy: ".")
            .first?
            .capitalized ?? "Discovered Feed"
    }

    private static func linkTags(in html: String) -> [String] {
        matches(in: html, pattern: "(?is)<link\\b[^>]*>")
    }

    private static func attributes(in tag: String) -> [String: String] {
        let pattern = #"([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*("([^"]*)"|'([^']*)'|([^\s"'=<>`]+))"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [:] }

        let range = NSRange(tag.startIndex..<tag.endIndex, in: tag)
        var output: [String: String] = [:]
        for match in regex.matches(in: tag, range: range) {
            guard let keyRange = Range(match.range(at: 1), in: tag) else { continue }
            let key = String(tag[keyRange]).lowercased()

            for captureIndex in [3, 4, 5] {
                guard match.range(at: captureIndex).location != NSNotFound,
                      let valueRange = Range(match.range(at: captureIndex), in: tag) else {
                    continue
                }
                output[key] = String(tag[valueRange])
                break
            }
        }
        return output
    }

    private static func firstMatch(in value: String, pattern: String) -> String? {
        matches(in: value, pattern: pattern).first
    }

    private static func matches(in value: String, pattern: String) -> [String] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(value.startIndex..<value.endIndex, in: value)
        return regex.matches(in: value, range: range).compactMap { match in
            let matchRange = match.numberOfRanges > 1 ? match.range(at: 1) : match.range
            guard let range = Range(matchRange, in: value) else { return nil }
            return String(value[range])
        }
    }

    private static func htmlDecoded(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
    }

    private static func unique(_ feeds: [DiscoveredFeed]) -> [DiscoveredFeed] {
        var seen = Set<String>()
        return feeds.filter { feed in
            seen.insert(feed.url.lowercased()).inserted
        }
    }
}
