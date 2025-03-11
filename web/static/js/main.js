// Main JavaScript for EV Opinion Search Engine

$(document).ready(function() {
    // Toggle advanced filters
    $('#toggle-filters').click(function() {
        $('#advanced-filters').slideToggle();
    });

    // Handle search form submission
    $('#search-form').submit(function(e) {
        e.preventDefault();
        var query = $('#search-input').val() || '*:*';
        performSearch(query, 0);
    });

    // Handle search input for suggestions
    $('#search-input').on('input', function() {
        var query = $(this).val();
        if (query.length >= 3) {
            $.getJSON('/suggest', {q: query}, function(data) {
                if (data.length > 0) {
                    var suggestions = '';
                    data.forEach(function(suggestion) {
                        suggestions += '<a href="#" class="list-group-item list-group-item-action suggestion-item">' + suggestion + '</a>';
                    });
                    $('#suggestions').html(suggestions).show();

                    // Handle suggestion click
                    $('.suggestion-item').click(function(e) {
                        e.preventDefault();
                        $('#search-input').val($(this).text());
                        $('#suggestions').hide();
                        $('#search-form').submit();
                    });
                } else {
                    $('#suggestions').hide();
                }
            });
        } else {
            $('#suggestions').hide();
        }
    });

    // Hide suggestions when clicking elsewhere
    $(document).click(function(e) {
        if (!$(e.target).closest('#search-input, #suggestions').length) {
            $('#suggestions').hide();
        }
    });

    // Load initial dashboard data
    loadDashboard();
});

// Load dashboard data and visualizations
function loadDashboard() {
    $.getJSON('/stats', function(data) {
        if (data.error) {
            $('#dashboard').html('<div class="alert alert-danger">' + data.error + '</div>');
            return;
        }

        // Update stats
        if (data.total_documents) {
            $('#document-count').text(data.total_documents.toLocaleString());
        }

        // Sentiment chart
        if (data.sentiment_distribution) {
            renderSentimentChart('sentiment-chart', data.sentiment_distribution);
        }

        // Get facet data for time chart
        $.getJSON('/facets', function(facetData) {
            if (facetData && facetData.facet_ranges && facetData.facet_ranges.created_utc) {
                var timeData = {};
                var counts = facetData.facet_ranges.created_utc.counts;
                for (var i = 0; i < counts.length; i += 2) {
                    timeData[counts[i]] = counts[i+1];
                }
                renderTimeChart('time-chart', timeData);
            }
        });
    });
}

// Perform search and update results
function performSearch(query, start) {
    // Show loading indicator
    $('#results').html('<div class="text-center my-5"><div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div></div>');

    // Hide dashboard, show results container
    $('#dashboard').hide();
    $('#search-results-container').show();

    // Get form data
    var formData = $('#search-form').serialize();
    formData += '&start=' + (start || 0);
    formData += '&rows=10';  // Number of results per page

    // Execute search
    $.getJSON('/search?' + formData, function(data) {
        if (data.error) {
            $('#results').html('<div class="alert alert-danger">' + data.error + '</div>');
            return;
        }

        // Update results info
        var resultsInfo = '<p>Found <strong>' + data.num_found.toLocaleString() + '</strong> results';
        if (query && query !== '*:*') {
            resultsInfo += ' for <strong>' + query + '</strong>';
        }
        resultsInfo += '</p>';
        $('#results-info').html(resultsInfo);

        // Render results
        renderSearchResults(data);

        // Render pagination
        renderPagination(data.num_found, data.start, 10, query);

        // Render facets
        if (data.facets) {
            renderFacets(data.facets);

            // Render visualizations for search results
            if (data.sentiment_chart) {
                $('#results-sentiment-chart').html('<img src="data:image/png;base64,' + data.sentiment_chart + '" class="img-fluid" alt="Sentiment Distribution">');
            }

            if (data.time_chart) {
                $('#time-chart').html('<img src="data:image/png;base64,' + data.time_chart + '" class="img-fluid" alt="Opinion Timeline">');
            }

            if (data.wordcloud) {
                $('#wordcloud').html('<img src="data:image/png;base64,' + data.wordcloud + '" class="img-fluid" alt="Word Cloud">');
            }
        }
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
        } else if (doc.title && doc.title.length > 0) {
            displayTitle = doc.title;
        } else {
            displayTitle = 'Article';
        }

        // Format the date
        const date = doc.created_utc ? new Date(doc.created_utc).toLocaleDateString() : 'Unknown date';

        // Determine sentiment class
        const sentimentClass = doc.sentiment ? `sentiment-${doc.sentiment.toLowerCase()}` : '';

        // Platform indicator - show News or other source icon
        const platformIcon = doc.platform === 'news' ?
                            '<i class="bi bi-newspaper text-info"></i>' :
                            '<i class="bi bi-chat-dots text-secondary"></i>';

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
                        ${doc.source_name ? `Source: ${doc.source_name} • ` : ''}
                        ${doc.author && doc.author !== 'Unknown' ? `By ${doc.author} • ` : ''}
                        ${date}
                    </small>
                    <span class="badge ${sentimentClass}">${doc.sentiment || 'Unknown'}</span>
                </div>
                
                ${renderEntityTopicTags(doc)}
                ${renderNewsMetadata(doc)}
            </div>
        `;

        results.append(resultHtml);
    });
}

/**
 * Render entity and topic tags
 */
function renderEntityTopicTags(doc) {
    let tagsHtml = '';

    // Add entities if present
    if (doc.entities && doc.entities.length > 0) {
        const entities = Array.isArray(doc.entities) ? doc.entities : [doc.entities];
        if (entities.length > 0) {
            tagsHtml += '<div class="mt-2">';
            tagsHtml += '<small class="text-muted me-2">Entities:</small>';

            entities.forEach(entity => {
                if (entity && entity.trim()) {
                    tagsHtml += `<span class="badge bg-light text-dark me-1">${entity}</span>`;
                }
            });

            tagsHtml += '</div>';
        }
    }

    // Add topics if present
    if (doc.topics && doc.topics.length > 0) {
        const topics = Array.isArray(doc.topics) ? doc.topics : [doc.topics];
        if (topics.length > 0) {
            tagsHtml += '<div class="mt-2">';
            tagsHtml += '<small class="text-muted me-2">Topics:</small>';

            topics.forEach(topic => {
                if (topic && topic.trim()) {
                    tagsHtml += `<span class="badge bg-secondary text-white me-1">${topic}</span>`;
                }
            });

            tagsHtml += '</div>';
        }
    }

    return tagsHtml;
}

/**
 * Render news article specific metadata
 */
function renderNewsMetadata(doc) {
    // Only render for news content
    if (doc.platform !== 'news' && doc.type !== 'news') {
        return '';
    }

    let newsHtml = '';

    // Add a link to the original article
    if (doc.url) {
        newsHtml += '<div class="mt-2">';
        newsHtml += `<a href="${doc.url}" target="_blank" class="btn btn-sm btn-outline-primary">
                        <i class="bi bi-box-arrow-up-right"></i> Read full article
                     </a>`;
        newsHtml += '</div>';
    }

    return newsHtml;
}

/**
 * Render facets for filtering
 */
function renderFacets(facets) {
    // Render topic facets
    if (facets.facet_fields && facets.facet_fields.topics) {
        renderFacetList('topic-facets', facets.facet_fields.topics, 'topic');
    }

    // Render entity facets
    if (facets.facet_fields && facets.facet_fields.entities) {
        renderFacetList('entity-facets', facets.facet_fields.entities, 'entity');
    }

    // Render source facets
    if (facets.facet_fields && facets.facet_fields.source_name) {
        renderFacetList('subreddit-facets', facets.facet_fields.source_name, 'subreddit');
    } else if (facets.facet_fields && facets.facet_fields.subreddit) {
        renderFacetList('subreddit-facets', facets.facet_fields.subreddit, 'subreddit');
    }
}

/**
 * Render a facet list
 */
function renderFacetList(elementId, facetData, paramName) {
    const container = $(`#${elementId}`);
    container.empty();

    // Convert facet data to object
    const facets = {};
    for (let i = 0; i < facetData.length; i += 2) {
        if (facetData[i] && facetData[i+1] > 0) {
            facets[facetData[i]] = facetData[i+1];
        }
    }

    // Sort facets by count (descending)
    const sortedFacets = Object.entries(facets)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);  // Limit to top 10

    // Create facet checkboxes
    sortedFacets.forEach(([facet, count]) => {
        // Get current URL parameters
        const urlParams = new URLSearchParams(window.location.search);
        const isChecked = urlParams.get(paramName) === facet;

        const checkboxHtml = `
            <div class="form-check">
                <input class="form-check-input facet-checkbox" type="checkbox" 
                       id="${paramName}-${facet}" name="${paramName}" value="${facet}" 
                       ${isChecked ? 'checked' : ''}>
                <label class="form-check-label" for="${paramName}-${facet}">
                    ${facet} <span class="badge rounded-pill bg-light text-dark">${count}</span>
                </label>
            </div>
        `;
        container.append(checkboxHtml);
    });

    // Add event listeners to facet checkboxes
    $('.facet-checkbox').change(function() {
        const name = $(this).attr('name');
        const value = $(this).val();

        // Update URL parameter
        const urlParams = new URLSearchParams(window.location.search);
        if ($(this).is(':checked')) {
            urlParams.set(name, value);
        } else {
            urlParams.delete(name);
        }

        // Update search with new parameters
        const query = urlParams.get('q') || '*:*';
        performSearch(query, 0);
    });
}

/**
 * Render pagination controls
 */
function renderPagination(total, start, rows, query) {
    const pagination = $('#pagination');
    pagination.empty();

    if (total <= rows) {
        return;  // No pagination needed
    }

    const totalPages = Math.ceil(total / rows);
    const currentPage = Math.floor(start / rows) + 1;

    let paginationHtml = '<ul class="pagination">';

    // Previous button
    if (currentPage > 1) {
        paginationHtml += `<li class="page-item">
            <a class="page-link" href="#" data-page="${currentPage - 1}">Previous</a>
        </li>`;
    } else {
        paginationHtml += `<li class="page-item disabled">
            <a class="page-link" href="#" tabindex="-1">Previous</a>
        </li>`;
    }

    // Page numbers
    const maxPages = 5;  // Maximum number of page links to show
    const halfMaxPages = Math.floor(maxPages / 2);

    let startPage = Math.max(1, currentPage - halfMaxPages);
    let endPage = Math.min(totalPages, startPage + maxPages - 1);

    if (endPage - startPage + 1 < maxPages) {
        startPage = Math.max(1, endPage - maxPages + 1);
    }

    for (let i = startPage; i <= endPage; i++) {
        if (i === currentPage) {
            paginationHtml += `<li class="page-item active">
                <a class="page-link" href="#">${i}</a>
            </li>`;
        } else {
            paginationHtml += `<li class="page-item">
                <a class="page-link" href="#" data-page="${i}">${i}</a>
            </li>`;
        }
    }

    // Next button
    if (currentPage < totalPages) {
        paginationHtml += `<li class="page-item">
            <a class="page-link" href="#" data-page="${currentPage + 1}">Next</a>
        </li>`;
    } else {
        paginationHtml += `<li class="page-item disabled">
            <a class="page-link" href="#" tabindex="-1">Next</a>
        </li>`;
    }

    paginationHtml += '</ul>';
    pagination.html(paginationHtml);

    // Add event listeners to pagination links
    $('.page-link').click(function(e) {
        e.preventDefault();
        const page = $(this).data('page');
        if (page) {
            const start = (page - 1) * rows;
            performSearch(query, start);
            // Scroll to top of results
            $('html, body').animate({
                scrollTop: $('#search-results-container').offset().top - 100
            }, 200);
        }
    });
}

/**
 * Render sentiment distribution chart
 */
function renderSentimentChart(elementId, data) {
    // This is a placeholder for a real chart implementation
    // In a real implementation, you would use a library like Chart.js

    const container = $(`#${elementId}`);
    container.empty();

    // Simple visual representation
    let chartHtml = '<div class="d-flex align-items-center justify-content-center">';

    // Define colors and labels
    const colors = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#6c757d'
    };

    // Calculate total for percentages
    const total = Object.values(data).reduce((sum, count) => sum + count, 0);

    // Create bars for each sentiment
    Object.entries(data).forEach(([sentiment, count]) => {
        const percentage = total > 0 ? (count / total * 100).toFixed(1) : 0;

        chartHtml += `
            <div class="mx-2 text-center">
                <div class="progress" style="width: 30px; height: 100px; transform: rotate(180deg);">
                    <div class="progress-bar" role="progressbar" style="width: 100%; height: ${percentage}%; background-color: ${colors[sentiment] || '#007bff'};"></div>
                </div>
                <div class="mt-2">
                    <span class="badge bg-${sentiment === 'positive' ? 'success' : sentiment === 'negative' ? 'danger' : 'secondary'}">${sentiment}</span>
                    <div>${percentage}%</div>
                </div>
            </div>
        `;
    });

    chartHtml += '</div>';
    container.html(chartHtml);
}

/**
 * Render time series chart
 */
function renderTimeChart(elementId, data) {
    // This is a placeholder for a real chart implementation
    // In a real implementation, you would use a library like Chart.js

    const container = $(`#${elementId}`);
    container.empty();

    // Create a simple representation
    const chartHtml = `
        <div class="alert alert-info">
            <i class="bi bi-info-circle"></i> 
            Data available for ${Object.keys(data).length} time periods.
        </div>
        <p class="text-center">Interactive timeline chart would be displayed here.</p>
    `;

    container.html(chartHtml);
}