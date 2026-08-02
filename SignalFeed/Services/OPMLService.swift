import Foundation

enum OPMLService {
    static func export(sources: [FeedSource]) -> String {
        let sortedSources = sources.sorted { $0.name < $1.name }
        var lines: [String] = [
            #"<?xml version="1.0" encoding="UTF-8"?>"#,
            #"<opml version="2.0">"#,
            "  <head>",
            "    <title>Signal Feed Sources</title>",
            "    <dateCreated>\(ISO8601DateFormatter().string(from: Date()))</dateCreated>",
            "  </head>",
            "  <body>"
        ]

        for source in sortedSources {
            let text = escaped(source.name)
            let xmlURL = escaped(source.url)
            let kind = escaped(source.kind.rawValue)
            let reputation = source.reputation.formatted(.number.precision(.fractionLength(2)))
            lines.append("    <outline text=\"\(text)\" title=\"\(text)\" type=\"rss\" xmlUrl=\"\(xmlURL)\" category=\"\(kind)\" signalReputation=\"\(reputation)\" />")
        }

        lines.append(contentsOf: [
            "  </body>",
            "</opml>"
        ])

        return lines.joined(separator: "\n")
    }

    static func parse(data: Data) throws -> [DiscoveredFeed] {
        let delegate = OPMLParserDelegate()
        let parser = XMLParser(data: data)
        parser.delegate = delegate
        parser.shouldResolveExternalEntities = false

        guard parser.parse() else {
            throw parser.parserError ?? OPMLServiceError.invalidDocument
        }

        return delegate.feeds
    }

    private static func escaped(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&#39;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }
}

enum OPMLServiceError: LocalizedError {
    case invalidDocument

    var errorDescription: String? {
        switch self {
        case .invalidDocument: "The OPML file could not be parsed."
        }
    }
}

private final class OPMLParserDelegate: NSObject, XMLParserDelegate {
    private(set) var feeds: [DiscoveredFeed] = []

    func parser(
        _ parser: XMLParser,
        didStartElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?,
        attributes attributeDict: [String: String] = [:]
    ) {
        guard elementName.lowercased() == "outline" else { return }
        let attrs = Dictionary(uniqueKeysWithValues: attributeDict.map { ($0.key.lowercased(), $0.value) })

        guard let xmlURL = attrs["xmlurl"]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !xmlURL.isEmpty else {
            return
        }

        let rawName = attrs["title"] ?? attrs["text"] ?? URL(string: xmlURL)?.host ?? "Imported Feed"
        feeds.append(
            DiscoveredFeed(
                title: rawName.collapsedWhitespace(),
                url: xmlURL,
                websiteURL: attrs["htmlurl"] ?? ""
            )
        )
    }
}
