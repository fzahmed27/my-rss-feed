import Foundation

struct RSSFeedParser {
    static func parse(data: Data, source: FeedSource) -> [FeedArticle] {
        let delegate = FeedParserDelegate(source: source)
        let parser = XMLParser(data: data)
        parser.delegate = delegate
        parser.shouldResolveExternalEntities = false
        parser.parse()
        return delegate.articles
    }
}

private final class FeedParserDelegate: NSObject, XMLParserDelegate {
    private struct Entry {
        var title = ""
        var link = ""
        var summary = ""
        var content = ""
        var published = ""
    }

    private let source: FeedSource
    private var currentEntry: Entry?
    private var currentElement = ""
    private var textBuffer = ""
    private var isInsideEntry = false

    private(set) var articles: [FeedArticle] = []

    init(source: FeedSource) {
        self.source = source
    }

    func parser(
        _ parser: XMLParser,
        didStartElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?,
        attributes attributeDict: [String: String] = [:]
    ) {
        let name = normalized(elementName)

        if name == "item" || name == "entry" {
            currentEntry = Entry()
            isInsideEntry = true
        }

        guard isInsideEntry else { return }

        currentElement = name
        textBuffer = ""

        if name == "link", let href = attributeDict["href"] {
            let rel = attributeDict["rel"] ?? "alternate"
            if rel == "alternate" || currentEntry?.link.isEmpty == true {
                currentEntry?.link = href
            }
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard isInsideEntry else { return }
        textBuffer += string
    }

    func parser(
        _ parser: XMLParser,
        didEndElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?
    ) {
        let name = normalized(elementName)

        if name == "item" || name == "entry" {
            finishEntry()
            currentEntry = nil
            isInsideEntry = false
            currentElement = ""
            textBuffer = ""
            return
        }

        guard isInsideEntry, name == currentElement else { return }

        let value = textBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }

        switch name {
        case "title":
            updateCurrentEntry { entry in
                entry.title = append(value, to: entry.title)
            }
        case "link":
            if currentEntry?.link.isEmpty == true {
                updateCurrentEntry { entry in
                    entry.link = value
                }
            }
        case "description", "summary", "subtitle":
            updateCurrentEntry { entry in
                entry.summary = append(value, to: entry.summary)
            }
        case "content", "encoded", "content:encoded":
            updateCurrentEntry { entry in
                entry.content = append(value, to: entry.content)
            }
        case "published", "updated", "created", "pubdate", "date", "dc:date":
            if currentEntry?.published.isEmpty == true {
                updateCurrentEntry { entry in
                    entry.published = value
                }
            }
        default:
            break
        }

        textBuffer = ""
    }

    private func finishEntry() {
        guard let entry = currentEntry else { return }
        let title = entry.title.removingHTML().collapsedWhitespace()
        let link = entry.link.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !title.isEmpty, !link.isEmpty else { return }

        let summary = entry.summary.removingHTML().collapsedWhitespace()
        let content = (entry.content.isEmpty ? entry.summary : entry.content)
            .removingHTML()
            .collapsedWhitespace()
        let id = RankingEngine.canonicalURL(from: link).isEmpty
            ? "\(source.id)-\(title.hashValue)"
            : RankingEngine.canonicalURL(from: link)

        articles.append(
            FeedArticle(
                id: id,
                sourceID: source.id,
                sourceName: source.name,
                sourceReputation: source.reputation,
                sourceKind: source.kind,
                title: title,
                link: link,
                summary: summary,
                content: content,
                publishedAt: DateParser.parse(entry.published)
            )
        )
    }

    private func normalized(_ elementName: String) -> String {
        elementName.lowercased()
    }

    private func updateCurrentEntry(_ update: (inout Entry) -> Void) {
        guard var entry = currentEntry else { return }
        update(&entry)
        currentEntry = entry
    }

    private func append(_ value: String, to existing: String?) -> String {
        guard let existing, !existing.isEmpty else { return value }
        return "\(existing) \(value)"
    }
}

enum DateParser {
    static func parse(_ rawValue: String) -> Date? {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let date = isoFormatter.date(from: trimmed) {
            return date
        }

        if let date = fractionalISOFormatter.date(from: trimmed) {
            return date
        }

        for format in rfcFormats {
            formatter.dateFormat = format
            if let date = formatter.date(from: trimmed) {
                return date
            }
        }

        return nil
    }

    private static let isoFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private static let fractionalISOFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let formatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.isLenient = true
        return formatter
    }()

    private static let rfcFormats = [
        "EEE, d MMM yyyy HH:mm:ss Z",
        "EEE, dd MMM yyyy HH:mm:ss Z",
        "d MMM yyyy HH:mm:ss Z",
        "yyyy-MM-dd'T'HH:mm:ssZ",
        "yyyy-MM-dd HH:mm:ss Z"
    ]
}

extension String {
    func removingHTML() -> String {
        let withoutScripts = replacingOccurrences(
            of: "<script[^>]*?>.*?</script>",
            with: " ",
            options: [.regularExpression, .caseInsensitive]
        )
        let withoutStyles = withoutScripts.replacingOccurrences(
            of: "<style[^>]*?>.*?</style>",
            with: " ",
            options: [.regularExpression, .caseInsensitive]
        )
        let withoutTags = withoutStyles.replacingOccurrences(
            of: "<[^>]+>",
            with: " ",
            options: [.regularExpression, .caseInsensitive]
        )

        return withoutTags
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
    }

    func collapsedWhitespace() -> String {
        components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }
}
