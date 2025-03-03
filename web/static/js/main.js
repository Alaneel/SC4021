/**
 * Main JavaScript for EV Opinion Search Engine
 */

$(document).ready(function() {
    // Initialize by loading overview stats
    loadDashboardStats();

    // Advanced filters toggle
    $('#toggle-filters').on('click', function() {
        $('#advanced-filters').slideToggle();

        // Toggle icon and text
        const $this = $(this);
        if ($this.find('i').hasClass('bi-sliders')) {
            $this.html('<i class="bi bi-chevron-up"></i> Hide Filters');
        } else {
            $this.html('<i class="bi bi-sliders"></i> Advanced Filters');
        }
    });

    // Search form submission
    $('#search-form').on('submit', function(e) {
        e.preventDefault();
        const query = $('#search-input').val() || '*:*';
        performSearch(query);
    });

    // Auto-suggest functionality
    let suggestionTimeout;
    $('#search-input').on('input', function() {
        const query = $(this).val();

        // Clear previous timeout
        clearTimeout(suggestionTimeout);

        if (query.length >= 2) {
            // Set a slight delay to avoid too many requests
            suggestionTimeout = setTimeout(function() {
                $.getJSON(`/suggest?q=${encodeURIComponent(query)}`, function(data) {
                    const suggestions = $('#suggestions');
                    suggestions.empty();

                    if (data.length > 0) {
                        data.forEach(suggestion => {
                            // Highlight the matching part
                            const highlightedSuggestion = suggestion.replace(
                                new RegExp(query, 'gi'),
                                match => `<strong>${match}</strong>`
                            );
                            suggestions.append(`<a class="list-group-item list-group-item-action">${highlightedSuggestion}</a>`);
                        });
                        suggestions.show();
                    } else {
                        suggestions.hide();
                    }
                });
            }, 300);
        } else {
            $('#suggestions').hide();
        }
    });

    // Handle suggestion click
    $(document).on('click', '#suggestions a', function() {
        const text = $(this).text();
        $('#search-input').val(text);
        $('#search-form').submit();
        $('#suggestions').hide();
    });

    // Hide suggestions when clicking elsewhere
    $(document).on('click', function(e) {
        if (!$(e.target).closest('#search-input, #suggestions').length) {
            $('#suggestions').hide();
        }
    });

    // Filter changes
    $('#sentiment-filter, #type-filter, #from-date, #to-date').on('change', function() {
        // Only trigger search if we have a query
        if ($('#search-input').val() || $('#results').children().length > 0) {
            $('#search-form').submit();
        }
    });

    // Facet click handler
    $(document).on('click', '.facet-value', function() {
        const facetField = $(this).data('field');
        const facetValue = $(this).data('value');

        // Update form with facet filter
        if (facetField === 'sentiment') {
            $('#sentiment-filter').val(facetValue).trigger('change');
        } else if (facetField === 'subreddit') {
            // Add or update hidden input
            if ($(`input[name="subreddit"]`).length) {
                $(`input[name="subreddit"]`).val(facetValue);
            } else {
                $('#search-form').append(`<input type="hidden" name="subreddit" value="${facetValue}">`);
            }
            $('#search-form').submit();
        } else if (facetField === 'topics') {
            // Add or update hidden input
            if ($(`input[name="topic"]`).length) {
                $(`input[name="topic"]`).val(facetValue);
            } else {
                $('#search-form').append(`<input type="hidden" name="topic" value="${facetValue}">`);
            }
            $('#search-form').submit();
        } else if (facetField === 'entities') {
            // Add or update hidden input
            if ($(`input[name="entity"]`).length) {
                $(`input[name="entity"]`).val(facetValue);
            } else {
                $('#search-form').append(`<input type="hidden" name="entity" value="${facetValue}">`);
            }
            $('#search-form').submit();
        } else if (facetField === 'type') {
            $('#type-filter').val(facetValue).trigger('change');
        }
    });

    // Clear all filters button
    $(document).on('click', '#clear-filters', function() {
        // Reset all form elements
        $('#search-form')[0].reset();
        $('#search-form input[type="hidden"]').remove();

        // Trigger search
        $('#search-form').submit();
    });
});

/**
 * Load overview statistics for the dashboard
 */
function loadDashboardStats() {
    $.getJSON('/stats', function(data) {
        // Update sentiment chart on the dashboard
        if (data.sentiment_distribution) {
            updateSentimentChart('#sentiment-chart', data.sentiment_distribution);
        }

        // Load facets for time chart
        $.getJSON('/facets', function(facetData) {
            if (facetData.facet_ranges && facetData.facet_ranges.created_utc) {
                const timeData = facetData.facet_ranges.created_utc.counts;
                const timeDict = {};

                // Convert to object with date -> count pairs
                for (let i = 0; i < timeData.length; i += 2) {
                    timeDict[timeData[i]] = timeData[i + 1];
                }

                updateTimeChart('#time-chart', timeDict);
            }
        });
    });
}

/**
 * Perform search with the given query
 */
function performSearch(query) {
    // Show loading
    $('#results').html('<div class="text-center p-5"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-3">Searching...</p></div>');

    // Show results container
    $('#search-results-container').show();

    // Hide dashboard if visible
    $('#dashboard').hide();

    // Get form data
    const formData = $('#search-form').serialize();

    // Execute search
    $.getJSON(`/search?${formData}`, function(data) {
        // Update results info
        $('#results-info').html(`
            <div class="d-flex justify-content-between align-items-center">
                <h4>Found ${data.num_found} results for: <span class="text-primary">${data.query !== '*:*' ? data.query : 'all opinions'}</span></h4>
                <button id="clear-filters" class="btn btn-outline-secondary btn-sm">
                    <i class="bi bi-x-circle"></i> Clear Filters
                </button>
            </div>
        `);

        // Render results
        renderSearchResults(data);

        // Update charts
        if (data.sentiment_chart) {
            $('#results-sentiment-chart').html(`<img src="data:image/png;base64,${data.sentiment_chart}" class="img-fluid" />`);
        }

        if (data.time_chart) {
            $('#time-chart').html(`<img src="data:image/png;base64,${data.time_chart}" class="img-fluid" />`);
        }

        if (data.wordcloud) {
            $('#wordcloud').html(`<img src="data:image/png;base64,${data.wordcloud}" class="img-fluid" />`);
        }

        // Update facets
        updateFacets(data.facets);

        // Render pagination
        renderPagination(data.start, data.rows, data.num_found);
    }).fail(function() {
        $('#results').html('<div class="alert alert-danger">Error performing search. Please try again later.</div>');
    });
}

/**
 * Render search results
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
        } else if (doc.title) {
            displayTitle = doc.title;
        } else {
            displayTitle = 'Comment';
        }

        // Format the date
        const date = doc.created_utc ? new Date(doc.created_utc).toLocaleDateString() : 'Unknown date';

        // Determine sentiment class
        const sentimentClass = doc.sentiment ? `sentiment-${doc.sentiment.toLowerCase()}` : '';

        // Create result item
        const resultHtml = `
            <div class="result-item">
                <h5>${displayTitle}</h5>
                <p class="mb-1">${displayText}</p>
                <div class="d-flex justify-content-between align-items-center mt-2">
                    <small class="text-muted">
                        Posted by ${doc.author || 'anonymous'} in r/${doc.subreddit || 'unknown'} on ${date}
                    </small>
                    <span class="badge ${sentimentClass}">${doc.sentiment || 'Unknown'}</span>
                </div>
                
                ${renderEntityTopicTags(doc)}
            </div>
        `;

        results.append(resultHtml);
    });
}

/**
 * Render entity and topic tags for a result
 */
function renderEntityTopicTags(doc) {
    let tagsHtml = '';

    // Add entities if present
    if (doc.entities && doc.entities.length > 0) {
        const entities = Array.isArray(doc.entities) ? doc.entities : [doc.entities];
        if (entities.length > 0) {
            tagsHtml += '<div class="mt-2">';
            tagsHtml += '<small class="text-muted me-2">Entities:</small>';

            entities.slice(0, 5).forEach(entity => {
                tagsHtml += `<span class="badge bg-light text-dark me-1 facet-value" data-field="entities" data-value="${entity}">${entity}</span>`;
            });

            if (entities.length > 5) {
                tagsHtml += `<span class="badge bg-light text-dark">+${entities.length - 5} more</span>`;
            }

            tagsHtml += '</div>';
        }
    }

    // Add topics if present
    if (doc.topics && doc.topics.length > 0) {
        const topics = Array.isArray(doc.topics) ? doc.topics : [doc.topics];
        if (topics.length > 0) {
            tagsHtml += '<div class="mt-1">';
            tagsHtml += '<small class="text-muted me-2">Topics:</small>';

            topics.slice(0, 3).forEach(topic => {
                tagsHtml += `<span class="badge bg-light text-dark me-1 facet-value" data-field="topics" data-value="${topic}">${topic}</span>`;
            });

            if (topics.length > 3) {
                tagsHtml += `<span class="badge bg-light text-dark">+${topics.length - 3} more</span>`;
            }

            tagsHtml += '</div>';
        }
    }

    return tagsHtml;
}

/**
 * Update facet filters
 */
function updateFacets(facets) {
    if (!facets || !facets.facet_fields) return;

    // Update topic facets
    if (facets.facet_fields.topics) {
        const topicFacets = $('#topic-facets');
        topicFacets.empty();

        const topics = {};
        for (let i = 0; i < facets.facet_fields.topics.length; i += 2) {
            topics[facets.facet_fields.topics[i]] = facets.facet_fields.topics[i + 1];
        }

        // Sort by count and limit to top 10
        const sortedTopics = Object.entries(topics)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        if (sortedTopics.length > 0) {
            sortedTopics.forEach(([topic, count]) => {
                topicFacets.append(`
                    <div class="form-check">
                        <label class="form-check-label">
                            <input class="form-check-input facet-value" type="checkbox" 
                                data-field="topics" data-value="${topic}">
                            ${topic} <span class="badge rounded-pill">${count}</span>
                        </label>
                    </div>
                `);
            });
        } else {
            topicFacets.html('<p class="text-muted small">No topics available</p>');
        }
    }

    // Update entity facets
    if (facets.facet_fields.entities) {
        const entityFacets = $('#entity-facets');
        entityFacets.empty();

        const entities = {};
        for (let i = 0; i < facets.facet_fields.entities.length; i += 2) {
            entities[facets.facet_fields.entities[i]] = facets.facet_fields.entities[i + 1];
        }

        // Sort by count and limit to top 10
        const sortedEntities = Object.entries(entities)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        if (sortedEntities.length > 0) {
            sortedEntities.forEach(([entity, count]) => {
                entityFacets.append(`
                    <div class="form-check">
                        <label class="form-check-label">
                            <input class="form-check-input facet-value" type="checkbox" 
                                data-field="entities" data-value="${entity}">
                            ${entity} <span class="badge rounded-pill">${count}</span>
                        </label>
                    </div>
                `);
            });
        } else {
            entityFacets.html('<p class="text-muted small">No entities available</p>');
        }
    }

    // Update subreddit facets
    if (facets.facet_fields.subreddit) {
        const subredditFacets = $('#subreddit-facets');
        subredditFacets.empty();

        const subreddits = {};
        for (let i = 0; i < facets.facet_fields.subreddit.length; i += 2) {
            subreddits[facets.facet_fields.subreddit[i]] = facets.facet_fields.subreddit[i + 1];
        }

        // Sort by count
        const sortedSubreddits = Object.entries(subreddits)
            .sort((a, b) => b[1] - a[1]);

        if (sortedSubreddits.length > 0) {
            sortedSubreddits.forEach(([subreddit, count]) => {
                subredditFacets.append(`
                    <div class="form-check">
                        <label class="form-check-label">
                            <input class="form-check-input facet-value" type="checkbox" 
                                data-field="subreddit" data-value="${subreddit}">
                            r/${subreddit} <span class="badge rounded-pill">${count}</span>
                        </label>
                    </div>
                `);
            });
        } else {
            subredditFacets.html('<p class="text-muted small">No sources available</p>');
        }
    }
}

/**
 * Render pagination controls
 */
function renderPagination(start, rows, numFound) {
    const pagination = $('#pagination');
    pagination.empty();

    if (numFound <= rows) return;

    const totalPages = Math.ceil(numFound / rows);
    const currentPage = Math.floor(start / rows) + 1;

    let paginationHtml = '<nav><ul class="pagination">';

    // Previous button
    if (currentPage > 1) {
        paginationHtml += `
            <li class="page-item">
                <a class="page-link" href="#" onclick="changePage(${(currentPage - 2) * rows}); return false;">Previous</a>
            </li>
        `;
    } else {
        paginationHtml += '<li class="page-item disabled"><a class="page-link">Previous</a></li>';
    }

    // Page numbers
    const startPage = Math.max(1, currentPage - 2);
    const endPage = Math.min(totalPages, startPage + 4);

    for (let i = startPage; i <= endPage; i++) {
        if (i === currentPage) {
            paginationHtml += `<li class="page-item active"><a class="page-link">${i}</a></li>`;
        } else {
            paginationHtml += `
                <li class="page-item">
                    <a class="page-link" href="#" onclick="changePage(${(i - 1) * rows}); return false;">${i}</a>
                </li>
            `;
        }
    }

    // Next button
    if (currentPage < totalPages) {
        paginationHtml += `
            <li class="page-item">
                <a class="page-link" href="#" onclick="changePage(${currentPage * rows}); return false;">Next</a>
            </li>
        `;
    } else {
        paginationHtml += '<li class="page-item disabled"><a class="page-link">Next</a></li>';
    }

    paginationHtml += '</ul></nav>';
    pagination.html(paginationHtml);
}

/**
 * Change page in search results
 */
function changePage(start) {
    // Update the start parameter and resubmit
    if ($('input[name="start"]').length) {
        $('input[name="start"]').val(start);
    } else {
        $('#search-form').append(`<input type="hidden" name="start" value="${start}">`);
    }

    $('#search-form').submit();

    // Scroll to top of results
    $('html, body').animate({
        scrollTop: $('#results-info').offset().top - 20
    }, 200);
}

/**
 * Update sentiment distribution chart
 */
function updateSentimentChart(selector, data) {
    const container = $(selector);

    // Calculate percentages
    const total = Object.values(data).reduce((sum, count) => sum + count, 0);
    const sentimentData = [];

    // Define colors for each sentiment
    const colors = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#6c757d'
    };

    // Format data for display
    Object.entries(data).forEach(([sentiment, count]) => {
        const percentage = (count / total * 100).toFixed(1);
        const color = colors[sentiment] || '#0d6efd';

        sentimentData.push({
            sentiment: sentiment.charAt(0).toUpperCase() + sentiment.slice(1),
            count: count,
            percentage: percentage,
            color: color
        });
    });

    // Sort by count (largest first)
    sentimentData.sort((a, b) => b.count - a.count);

    // Create HTML for the chart
    let chartHtml = '<div class="sentiment-distribution">';

    // Add donut visualization
    chartHtml += '<div class="sentiment-donut position-relative" style="height: 150px;">';

    // Calculate stroke-dasharray values for the donut segments
    let cumulativePercentage = 0;
    sentimentData.forEach(item => {
        const percentage = parseFloat(item.percentage);

        chartHtml += `
            <svg viewBox="0 0 36 36" class="position-absolute top-0 start-0 w-100 h-100">
                <path d="M18 2.0845
                    a 15.9155 15.9155 0 0 1 0 31.831
                    a 15.9155 15.9155 0 0 1 0 -31.831"
                    fill="none" stroke="${item.color}" stroke-width="3"
                    stroke-dasharray="${percentage}, 100"
                    stroke-dashoffset="${-cumulativePercentage}"
                    stroke-linecap="round" />
            </svg>
        `;

        cumulativePercentage += percentage;
    });

    // Add center text with total count
    chartHtml += `
        <div class="position-absolute top-50 start-50 translate-middle text-center">
            <div class="fw-bold">${total}</div>
            <div class="small">opinions</div>
        </div>
    `;

    chartHtml += '</div>';

    // Add legend
    chartHtml += '<div class="sentiment-legend mt-3">';

    sentimentData.forEach(item => {
        chartHtml += `
            <div class="d-flex align-items-center mb-1">
                <div class="me-2" style="width: 12px; height: 12px; background-color: ${item.color}; border-radius: 2px;"></div>
                <div class="d-flex justify-content-between w-100">
                    <span>${item.sentiment}</span>
                    <span>${item.percentage}%</span>
                </div>
            </div>
        `;
    });

    chartHtml += '</div>';
    chartHtml += '</div>';

    // Set the HTML
    container.html(chartHtml);
}

/**
 * Update time chart with opinion counts over time
 */
function updateTimeChart(selector, timeData) {
    const container = $(selector);

    // Convert data for display
    const chartData = [];
    Object.entries(timeData).forEach(([dateStr, count]) => {
        const date = new Date(dateStr);
        chartData.push({
            date: date,
            dateStr: date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' }),
            count: count
        });
    });

    // Sort by date
    chartData.sort((a, b) => a.date - b.date);

    // Find max count for scaling
    const maxCount = Math.max(...chartData.map(d => d.count));
    const chartHeight = 150;

    // Generate SVG line chart
    let svgHtml = `
        <svg viewBox="0 0 ${chartData.length * 40} ${chartHeight + 30}" class="w-100" style="height: 180px;">
    `;

    // Add bars
    chartData.forEach((item, index) => {
        const barHeight = (item.count / maxCount) * chartHeight;
        const barWidth = 20;
        const x = index * 40 + 10; // 40px per bar, centered
        const y = chartHeight - barHeight;

        svgHtml += `
            <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" 
                  fill="#0d6efd" opacity="0.7" rx="2" />
                  
            <text x="${x + barWidth/2}" y="${y - 5}" 
                  text-anchor="middle" font-size="8" fill="#666">
                ${item.count}
            </text>
            
            <text x="${x + barWidth/2}" y="${chartHeight + 15}" 
                  text-anchor="middle" font-size="8" fill="#666">
                ${item.dateStr}
            </text>
        `;
    });

    svgHtml += '</svg>';

    // Set the HTML
    container.html(svgHtml);
}