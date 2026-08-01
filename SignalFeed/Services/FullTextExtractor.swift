import Foundation

enum FullTextExtractor {
    static func enrich(
        articles: [FeedArticle],
        limit: Int
    ) async -> [FeedArticle] {
        guard limit > 0 else { return articles }

        let selectedIDs = Set(
            articles
                .sorted { ($0.publishedAt ?? .distantPast) > ($1.publishedAt ?? .distantPast) }
                .prefix(limit)
                .map(\.id)
        )
        var extractedByID: [String: String] = [:]

        await withTaskGroup(of: (String, String?).self) { group in
            for article in articles where selectedIDs.contains(article.id) {
                group.addTask {
                    let extracted = await extract(from: article.link)
                    return (article.id, extracted)
                }
            }

            for await result in group {
                if let text = result.1, !text.isEmpty {
                    extractedByID[result.0] = text
                }
            }
        }

        guard !extractedByID.isEmpty else { return articles }
        return articles.map { article in
            if let extracted = extractedByID[article.id] {
                return article.withExtractedContent(extracted)
            }
            return article
        }
    }

    private static func extract(from rawURL: String) async -> String? {
        guard let url = URL(string: rawURL),
              ["http", "https"].contains(url.scheme?.lowercased() ?? "") else {
            return nil
        }

        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 12
            request.cachePolicy = .returnCacheDataElseLoad
            request.setValue("SignalFeed-iOS/1.0", forHTTPHeaderField: "User-Agent")
            request.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")

            let (data, response) = try await URLSession.shared.data(for: request)
            if let httpResponse = response as? HTTPURLResponse,
               !(200..<300).contains(httpResponse.statusCode) {
                return nil
            }

            if let mimeType = response.mimeType?.lowercased(),
               !mimeType.contains("html"),
               !mimeType.contains("text") {
                return nil
            }

            let capped = data.prefix(1_500_000)
            let html = String(data: capped, encoding: .utf8)
                ?? String(data: capped, encoding: .isoLatin1)
                ?? ""
            let text = readableText(from: html)
            return text.count > 400 ? String(text.prefix(18_000)) : nil
        } catch {
            return nil
        }
    }

    private static func readableText(from html: String) -> String {
        let cleaned = html
            .removingMatches("(?is)<script[^>]*>.*?</script>")
            .removingMatches("(?is)<style[^>]*>.*?</style>")
            .removingMatches("(?is)<noscript[^>]*>.*?</noscript>")
            .removingMatches("(?is)<svg[^>]*>.*?</svg>")
            .removingMatches("(?is)<nav[^>]*>.*?</nav>")
            .removingMatches("(?is)<header[^>]*>.*?</header>")
            .removingMatches("(?is)<footer[^>]*>.*?</footer>")
            .removingMatches("(?is)<aside[^>]*>.*?</aside>")
            .removingMatches("(?is)<form[^>]*>.*?</form>")

        if let articleBlock = firstMatch(in: cleaned, pattern: "(?is)<article[^>]*>(.*?)</article>") {
            let articleText = textFromHTMLBlock(articleBlock)
            if articleText.count > 400 {
                return articleText
            }
        }

        let paragraphs = matches(in: cleaned, pattern: "(?is)<p[^>]*>(.*?)</p>")
            .map(textFromHTMLBlock)
            .filter { $0.count > 40 }
        let paragraphText = paragraphs.joined(separator: " ")
        if paragraphText.count > 400 {
            return paragraphText.collapsedWhitespace()
        }

        if let body = firstMatch(in: cleaned, pattern: "(?is)<body[^>]*>(.*?)</body>") {
            let bodyText = textFromHTMLBlock(body)
            if bodyText.count > 400 {
                return bodyText
            }
        }

        return textFromHTMLBlock(cleaned)
    }

    private static func textFromHTMLBlock(_ block: String) -> String {
        block
            .replacingOccurrences(of: "(?is)<br\\s*/?>", with: " ", options: .regularExpression)
            .replacingOccurrences(of: "(?is)</(p|div|li|h1|h2|h3|h4|section)>", with: " ", options: .regularExpression)
            .removingHTML()
            .collapsedWhitespace()
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
}

private extension String {
    func removingMatches(_ pattern: String) -> String {
        replacingOccurrences(
            of: pattern,
            with: " ",
            options: [.regularExpression, .caseInsensitive]
        )
    }
}
