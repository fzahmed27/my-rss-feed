import Foundation

struct RSSFeedParser {
    static func parse(data: Data, source: FeedSource) -> [FeedArticle] {
        let delegate = FeedParserDelegate(source: source)
        let parser = XMLParser(data: data)
        parser.delegate = delegate
        parser.shouldProcessNamespaces = true
        parser.shouldResolveExternalEntities = false
        parser.parse()
        return delegate.articles
    }
}

private final class FeedParserDelegate: NSObject, XMLParserDelegate {
    private enum CapturedField {
        case title
        case link
        case summary
        case content
        case published
        case identifier
    }

    private struct Entry {
        var title = ""
        var link = ""
        var linkPriority = -1
        var summary = ""
        var content = ""
        var published = ""
        var identifier = ""
    }

    private let source: FeedSource
    private var currentEntry: Entry?
    private var capturedField: CapturedField?
    private var captureDepth = 0
    private var entryDepth = 0
    private var elementDepth = 0
    private var textBuffer = ""

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
        elementDepth += 1
        let name = normalized(elementName)

        if name == "item" || name == "entry" {
            currentEntry = Entry()
            entryDepth = elementDepth
            clearCapture()
            return
        }

        guard currentEntry != nil, capturedField == nil else { return }

        switch name {
        case "title":
            startCapture(.title)
        case "link":
            if let href = attributeValue(named: "href", in: attributeDict), !href.isEmpty {
                selectLink(href, relationship: attributeValue(named: "rel", in: attributeDict))
            } else {
                startCapture(.link)
            }
        case "description", "summary", "subtitle":
            startCapture(.summary)
        case "content", "encoded":
            startCapture(.content)
        case "published", "updated", "created", "pubdate", "date":
            if currentEntry?.published.isEmpty == true {
                startCapture(.published)
            }
        case "id", "guid":
            if currentEntry?.identifier.isEmpty == true {
                startCapture(.identifier)
            }
        default:
            break
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard capturedField != nil else { return }
        textBuffer += string
    }

    func parser(_ parser: XMLParser, foundCDATA CDATABlock: Data) {
        guard capturedField != nil,
              let value = String(data: CDATABlock, encoding: .utf8) else { return }
        textBuffer += value
    }

    func parser(
        _ parser: XMLParser,
        didEndElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?
    ) {
        defer { elementDepth -= 1 }
        let name = normalized(elementName)

        if let capturedField, elementDepth == captureDepth {
            commitCapturedValue(textBuffer, to: capturedField)
            clearCapture()
        }

        if (name == "item" || name == "entry"), elementDepth == entryDepth {
            finishEntry()
            currentEntry = nil
            entryDepth = 0
            clearCapture()
        }
    }

    private func startCapture(_ field: CapturedField) {
        capturedField = field
        captureDepth = elementDepth
        textBuffer = ""
    }

    private func clearCapture() {
        capturedField = nil
        captureDepth = 0
        textBuffer = ""
    }

    private func commitCapturedValue(_ rawValue: String, to field: CapturedField) {
        let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { return }

        updateCurrentEntry { entry in
            switch field {
            case .title:
                entry.title = append(value, to: entry.title)
            case .link:
                if entry.link.isEmpty {
                    entry.link = value
                    entry.linkPriority = 2
                }
            case .summary:
                entry.summary = append(value, to: entry.summary)
            case .content:
                entry.content = append(value, to: entry.content)
            case .published:
                if entry.published.isEmpty {
                    entry.published = value
                }
            case .identifier:
                if entry.identifier.isEmpty {
                    entry.identifier = value
                }
            }
        }
    }

    private func selectLink(_ href: String, relationship: String?) {
        let relation = relationship?.lowercased() ?? ""
        let priority = switch relation {
        case "alternate": 3
        case "": 2
        case "self": 0
        default: 1
        }

        updateCurrentEntry { entry in
            if priority > entry.linkPriority {
                entry.link = href
                entry.linkPriority = priority
            }
        }
    }

    private func finishEntry() {
        guard let entry = currentEntry else { return }
        let title = entry.title.removingHTML().collapsedWhitespace()
        let fallbackLink = entry.identifier.hasPrefix("http") ? entry.identifier : ""
        let link = (entry.link.isEmpty ? fallbackLink : entry.link)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        guard !title.isEmpty, !link.isEmpty else { return }

        let summary = entry.summary.removingHTML().collapsedWhitespace()
        let content = (entry.content.isEmpty ? entry.summary : entry.content)
            .removingHTML()
            .collapsedWhitespace()
        let canonicalURL = RankingEngine.canonicalURL(from: link)
        let id = canonicalURL.isEmpty ? "\(source.id):\(link.lowercased())" : canonicalURL

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
        elementName
            .split(separator: ":")
            .last?
            .lowercased() ?? elementName.lowercased()
    }

    private func attributeValue(named name: String, in attributes: [String: String]) -> String? {
        attributes.first { normalized($0.key) == name }?.value
    }

    private func updateCurrentEntry(_ update: (inout Entry) -> Void) {
        guard var entry = currentEntry else { return }
        update(&entry)
        currentEntry = entry
    }

    private func append(_ value: String, to existing: String) -> String {
        existing.isEmpty ? value : "\(existing) \(value)"
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
