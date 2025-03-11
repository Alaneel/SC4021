// Add these functions to web/static/js/main.js

/**
 * Render search results with X-specific content
 */
function renderSearchResults(data) {
    const results = $('#results');
    results.empty();

    if (data.docs.length === 0) {
        results.html('<div class="alert alert-info">No results found. Try adjusting your search criteria.</div>');
        return;
    }

    data.docs.forEach(doc => {
        const highlighting = data.highlighting && data.highlighting[doc.id] ? data.highlighting[doc.id] : {};

        // Use highlighted text if available, otherwise show a snippet
        let displayText = '';
        if (highlighting.text && highlighting.text.length > 0) {
            displayText = highlighting.text.join('... ');
        } else if (doc.text) {
            displayText = doc.text.length > 300 ? doc.text.substring(0, 300) + '...' : doc.text;
        }

        // Use highlighted title if available
        let displayTitle = '';
        if (highlighting.title && highlighting.title.length > 0) {
            displayTitle = highlighting.title.join(' ');
        } else if (doc.title && doc.title.length > 0) {
            displayTitle = doc.title;
        } else if (doc.type === 'tweet') {
            // For tweets with no title, use a default
            displayTitle = 'Tweet';
        } else {
            displayTitle = 'Comment';
        }

        // Format the date
        const date = doc.created_utc ? new Date(doc.created_utc).toLocaleDateString() : 'Unknown date';

        // Determine sentiment class
        const sentimentClass = doc.sentiment ? `sentiment-${doc.sentiment.toLowerCase()}` : '';

        // Platform indicator - show X or Reddit icon
        const platformIcon = doc.platform === 'x' || doc.type === 'tweet' ?
                            '<i class="bi bi-twitter text-info"></i>' :
                            '<i class="bi bi-reddit text-danger"></i>';

        // Create result item
        const resultHtml = `
            <div class="result-item">
                <div class="d-flex justify-content-between">
                    <h5>${displayTitle}</h5>
                    <span>${platformIcon}</span>
                </div>
                <p class="mb-1">${displayText}</p>
                <div class="d-flex justify-content-between align-items-center mt-2">
                    <small class="text-muted">
                        Posted by ${doc.author || 'anonymous'} on ${date}
                    </small>
                    <span class="badge ${sentimentClass}">${doc.sentiment || 'Unknown'}</span>
                </div>
                
                ${renderEntityTopicTags(doc)}
                ${renderXSpecificContent(doc)}
            </div>
        `;

        results.append(resultHtml);
    });
}

/**
 * Render X-specific content like hashtags and metrics
 */
function renderXSpecificContent(doc) {
    // Only render for X content
    if (doc.platform !== 'x' && doc.type !== 'tweet') {
        return '';
    }

    let xHtml = '';

    // Add hashtags if present
    if (doc.hashtags && doc.hashtags.length > 0) {
        const hashtags = Array.isArray(doc.hashtags) ? doc.hashtags : [doc.hashtags];
        if (hashtags.length > 0) {
            xHtml += '<div class="mt-2">';
            xHtml += '<small class="text-muted me-2">Hashtags:</small>';

            hashtags.forEach(hashtag => {
                if (hashtag && hashtag.trim()) {
                    xHtml += `<span class="badge bg-info text-dark me-1">#${hashtag}</span>`;
                }
            });

            xHtml += '</div>';
        }
    }

    // Add metrics if present (likes, retweets, etc.)
    if (doc.score || doc.retweet_count || doc.quote_count) {
        xHtml += '<div class="mt-2 d-flex gap-3">';

        if (doc.score) {
            xHtml += `<small class="text-muted"><i class="bi bi-heart"></i> ${doc.score}</small>`;
        }

        if (doc.retweet_count) {
            xHtml += `<small class="text-muted"><i class="bi bi-repeat"></i> ${doc.retweet_count}</small>`;
        }

        if (doc.quote_count) {
            xHtml += `<small class="text-muted"><i class="bi bi-chat-quote"></i> ${doc.quote_count}</small>`;
        }

        xHtml += '</div>';
    }

    return xHtml;
}