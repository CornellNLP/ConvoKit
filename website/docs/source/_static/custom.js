// docs/source/_static/custom.js

function getEffectiveTheme() {
    const html = document.documentElement;

    // 1. Explicit data-theme attribute (Furo, etc.)
    const dataTheme = html.getAttribute('data-theme');
    if (dataTheme === 'dark')  return 'dark';
    if (dataTheme === 'light') return 'light';

    // 2. Bootstrap-style data-bs-theme (PyData Sphinx Theme)
    const bsTheme = html.getAttribute('data-bs-theme');
    if (bsTheme === 'dark')  return 'dark';
    if (bsTheme === 'light') return 'light';

    // 3. Class-based (some themes add class "dark" or "theme-dark" to <html>)
    if (html.classList.contains('dark') || html.classList.contains('theme-dark')) return 'dark';
    if (html.classList.contains('light') || html.classList.contains('theme-light')) return 'light';

    // 4. Fall back to OS preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyThemeClass() {
    const theme = getEffectiveTheme();
    document.body.classList.toggle('convokit-dark', theme === 'dark');
}

// Apply immediately on load, then watch for any further changes
document.addEventListener('DOMContentLoaded', applyThemeClass);

// Watch <html> for attribute mutations (catches all Sphinx theme toggles)
new MutationObserver(applyThemeClass).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme', 'data-bs-theme', 'class']
});

// Also react if the OS preference changes while the page is open
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyThemeClass);

// ─────────────────────────────────────────────────────────────────────────────

// Tag type definitions for DATASET pages
const TAG_TYPES = {
    'Location': 'location',
    'Conversation Type': 'conversation-type',
    'Metadata': 'metadata',
    'Source': 'source',
    'Dataset Size': 'dataset-size',
    'Conversation Length': 'conversation-length',
    'Topics': 'topics',
    'Language': 'language',
    'Format': 'format',
    'Dynamics': 'dynamics'
};

// Tag type definitions for feature page
const FEATURE_TAG_TYPES = {
    'Analysis Type': 'analysis-type',
    'Analysis Method': 'analysis-method',
    'Analysis Level': 'analysis-level',
    'Analysis Focus': 'analysis-focus'
};

// tag sets for feature pages
const FEATURE_TAG_SETS = {
    'analysis-type': new Set([
        'prediction', 'classification', 'structural', 'sorting', 'measurement',
        'feature extraction', 'pre-processing'
    ]),
    'analysis-method': new Set([
        'statistical', 'modeling', 'graph', 'machine learning', 'neural', 'llm',
        'simulation', 'parsing', 'vectorization'
    ]),
    'analysis-level': new Set([
        'utterance', 'exchange', 'conversation', 'speaker', 'corpus'
    ]),
    'analysis-focus': new Set([
        'linguistic', 'power', 'influence', 'social', 'development', 'politeness',
        'context', 'pattern', 'diversity', 'conversation-flow', 'turning-points',
        'forecasting', 'detection', 'representation', 'labeling', 'comparison',
        'pragmatics'
    ])
};

document.addEventListener('DOMContentLoaded', function() {
    // Initialize dataset search and filters
    initializeSearch('dataset-search', '.dataset-card', 'dataset');
    initializeSearch('feature-search', '.feature-card', 'feature');
});

function initializeSearch(searchInputId, cardSelector, context) {
    const searchInput = document.getElementById(searchInputId);
    if (!searchInput) return;

    const cards = document.querySelectorAll(cardSelector);
    const container = searchInput.closest('.dataset-search-container, .feature-search-container');
    const tagFiltersContainer = container.querySelector('.tag-filters');
    const clearButton = container.querySelector('.clear-filters');

    let activeFilters = new Set();
    let allTags = new Map(); // Map of tag

    // Extract all tags from dataset cards
    extractAllTags();

    // Initial render
    renderTagFilters('');

    // Search functionality
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.trim();
        renderTagFilters(searchTerm);
        filterCards();
    });

    // Clear filters functionality
    if (clearButton) {
        clearButton.addEventListener('click', function() {
            activeFilters.clear();
            searchInput.value = '';
            renderTagFilters('');
            filterCards();
        });
    }

    function extractAllTags() {
        allTags.clear();

        cards.forEach(card => {
            const cardTags = card.dataset.tags ? card.dataset.tags.split(',').map(t => t.trim()) : [];

            cardTags.forEach(tag => {
                if (!tag) return;

                // Determine tag type
                const tagType = determineTagType(tag, context);

                if (!allTags.has(tag)) {
                    allTags.set(tag, { type: tagType, count: 0 });
                }
                allTags.get(tag).count++;
            });
        });
    }

    function determineTagType(tag, ctx) {
        const tagLower = tag.toLowerCase().trim();

        // Feature page context
        if (ctx === 'feature') {
            for (const [typeId, tagSet] of Object.entries(FEATURE_TAG_SETS)) {
                if (tagSet.has(tagLower)) return typeId;
            }
            // Default for unknown feature tags
            return 'analysis-focus';
        }

        // Dataset page context
        // Location tags
        if (tagLower.includes('in person') || tagLower.includes('online') ||
            tagLower.includes('fictional')){
            return 'location';
        }

        // Conversation type tags
        if (tagLower.includes('group') || tagLower.includes('dyadic') ||
            tagLower.includes('symmetric') || tagLower.includes('asymmetric') ||
            tagLower.includes('synchronous') || tagLower.includes('asynchronous')) {
            return 'conversation-type';
        }

        // Metadata tags
        if (tagLower.includes('outcome labels') || tagLower.includes('utterance labels') ||
            tagLower.includes('speaker info') || tagLower.includes('summaries') ||
            tagLower.includes('timestamps')) {
            return 'metadata';
        }

        // Source tags
        if (tagLower.includes('reddit') || tagLower.includes('wikipedia') ||
            tagLower.includes('twitter/x') || tagLower.includes('institutional') ||
            tagLower.includes('media') || tagLower.includes('stack exchange')) {
            return 'source';
        }

        // Dataset size tags
        if (tagLower.includes('small size') || tagLower.includes('medium size') ||
            tagLower.includes('large size')) {
            return 'dataset-size';
        }

        // Conversation length tags
        if (tagLower.includes('short conversations') || tagLower.includes('medium conversations') ||
            tagLower.includes('long conversations')) {
            return 'conversation-length';
        }

        // Topics tags
        if (tagLower.includes('politics') || tagLower.includes('law') ||
            tagLower.includes('movies') || tagLower.includes('sports') ||
            tagLower.includes('work') || tagLower.includes('various topics') ||
            tagLower.includes('financial')) {
            return 'topics';
        }

        // Language tags
        if (tagLower.includes('english') || tagLower.includes('multiple languages')) {
            return 'language';
        }

        // Format tags
        if (tagLower.includes('interviews') || tagLower.includes('collaboration') ||
            tagLower.includes('debate') || tagLower.includes('customer support')) {
            return 'format';
        }

        // Dynamics tags
        if (tagLower.includes('persuasion') || tagLower.includes('sarcasm') ||
            tagLower.includes('derailment') || tagLower.includes('recovery') ||
            tagLower.includes('negotiation') || tagLower.includes('q&a') ||
            tagLower.includes('politness') || tagLower.includes('deception') ||
            tagLower.includes('problem solving')) {
            return 'dynamics';
        }

        // Default to topics if unknown
        return 'topics';
    }

    function renderTagFilters(searchTerm) {
        const isSearching = searchTerm.length > 0;

        if (isSearching) {
            renderFlatTags(searchTerm);
        } else {
            renderHierarchicalTags();
        }
    }

    function renderFlatTags(searchTerm) {
        // Get matching tags based on search term
        const matchingTags = getMatchingTags(searchTerm);

        // Clear current filters
        tagFiltersContainer.innerHTML = '';

        if (matchingTags.size === 0) {
            tagFiltersContainer.innerHTML = '<div style="color: #7f8c8d; font-style: italic;">No matching tags found</div>';
            return;
        }

        // Create flat list of tags
        const tagsArray = Array.from(matchingTags.entries()).sort((a, b) => a[0].localeCompare(b[0]));

        tagsArray.forEach(([tag, info]) => {
            const button = createTagButton(tag, info.type);
            tagFiltersContainer.appendChild(button);
        });
    }

    function renderHierarchicalTags() {
        // Clear current filters
        tagFiltersContainer.innerHTML = '';

        // Create tag groups container
        const groupsContainer = document.createElement('div');
        groupsContainer.className = 'tag-groups';

        // Group tags by type
        const tagsByType = new Map();
        allTags.forEach((info, tag) => {
            if (!tagsByType.has(info.type)) {
                tagsByType.set(info.type, []);
            }
            tagsByType.get(info.type).push(tag);
        });

        // Create groups for each tag type that has tags
        const activeTagTypes = context === 'feature' ? FEATURE_TAG_TYPES : TAG_TYPES;
        Object.entries(activeTagTypes).forEach(([typeName, typeId]) => {
            const tagsOfType = tagsByType.get(typeId);
            if (!tagsOfType || tagsOfType.length === 0) return;

            const group = createTagGroup(typeName, typeId, tagsOfType);
            groupsContainer.appendChild(group);
        });

        tagFiltersContainer.appendChild(groupsContainer);
    }

    function createTagGroup(typeName, typeId, tags) {
        const group = document.createElement('div');
        group.className = 'tag-group';

        // Create header
        const header = document.createElement('div');
        header.className = 'tag-group-header collapsed';
        header.dataset.type = typeId;
        header.textContent = typeName;

        // Create tags container
        const tagsContainer = document.createElement('div');
        tagsContainer.className = 'tag-group-tags collapsed';

        // Sort and add tags
        tags.sort().forEach(tag => {
            const button = createTagButton(tag, typeId);
            tagsContainer.appendChild(button);
        });

        // Toggle functionality
        header.addEventListener('click', function() {
            const isCollapsed = this.classList.contains('collapsed');

            if (isCollapsed) {
                this.classList.remove('collapsed');
                this.classList.add('expanded');
                tagsContainer.classList.remove('collapsed');
            } else {
                this.classList.add('collapsed');
                this.classList.remove('expanded');
                tagsContainer.classList.add('collapsed');
            }
        });

        group.appendChild(header);
        group.appendChild(tagsContainer);

        return group;
    }

    function createTagButton(tag, type) {
        const button = document.createElement('button');
        button.className = 'tag-filter';
        button.dataset.tag = tag;
        button.dataset.type = type;
        button.textContent = tag;

        if (activeFilters.has(tag)) {
            button.classList.add('active');
        }

        button.addEventListener('click', function() {
            const tagName = this.dataset.tag;

            if (activeFilters.has(tagName)) {
                activeFilters.delete(tagName);
                this.classList.remove('active');
            } else {
                activeFilters.add(tagName);
                this.classList.add('active');
            }

            filterCards();
        });

        return button;
    }

    function getMatchingTags(searchTerm) {
        const matchingTags = new Map();
        const lowerSearch = searchTerm.toLowerCase();

        allTags.forEach((info, tag) => {
            if (tag.toLowerCase().includes(lowerSearch)) {
                matchingTags.set(tag, info);
            }
        });

        return matchingTags;
    }

    function filterCards() {
        const searchTerm = searchInput.value.toLowerCase();
        let visibleCount = 0;

        cards.forEach(card => {
            const text = card.textContent.toLowerCase();
            const cardTags = card.dataset.tags ? card.dataset.tags.toLowerCase() : '';

            // Check search term match
            const matchesSearch = !searchTerm ||
                                  text.includes(searchTerm) ||
                                  cardTags.includes(searchTerm);

            // Check tag filter match
            const matchesFilters = activeFilters.size === 0 ||
                                   Array.from(activeFilters).every(filter =>
                                       cardTags.includes(filter.toLowerCase())
                                   );

            if (matchesSearch && matchesFilters) {
                card.classList.remove('hidden');
                card.style.display = '';
                visibleCount++;

                // Show the associated header
                const header = findAssociatedHeader(card);
                if (header) {
                    header.style.display = '';
                }
            } else {
                card.classList.add('hidden');
                card.style.display = 'none';

                // Hide the associated header
                const header = findAssociatedHeader(card);
                if (header) {
                    header.style.display = 'none';
                }
            }
        });

        // Show "no results" message if needed
        updateNoResultsMessage(searchInput, visibleCount);
    }

    function findAssociatedHeader(card) {
        // Find the previous sibling that is a header (h2, h3, h4)
        let sibling = card.previousElementSibling;
        while (sibling) {
            if (sibling.tagName && sibling.tagName.match(/^H[2-4]$/)) {
                return sibling;
            }
            sibling = sibling.previousElementSibling;
        }
        return null;
    }

    function updateNoResultsMessage(input, count) {
        const container = input.closest('.dataset-search-container, .feature-search-container')
                                .nextElementSibling;

        // Remove existing message
        const existingMessage = container.querySelector('.no-results');
        if (existingMessage) {
            existingMessage.remove();
        }

        // Add message if no results
        if (count === 0) {
            const message = document.createElement('div');
            message.className = 'no-results';
            message.textContent = 'No results found. Try different search terms or filters.';
            container.insertBefore(message, container.firstChild);
        }
    }
}

// Add smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            e.preventDefault();
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Highlight active section in navigation
window.addEventListener('scroll', function() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.wy-menu-vertical a');

    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (pageYOffset >= sectionTop - 60) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('current');
        if (link.getAttribute('href') === '#' + current) {
            link.classList.add('current');
        }
    });
});
