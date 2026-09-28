import { useEffect, useMemo, useRef, useState } from 'react';
import { searchPlaces, searchRoads } from './places';

const SEARCH_DEBOUNCE_MS = 300;
const MIN_PLACE_QUERY_LENGTH = 3;

const FIELDS = {
  origin: {
    label: 'Starting point',
    placeholder: 'Choose starting point, or click the map',
  },
  destination: {
    label: 'Destination',
    placeholder: 'Choose destination, or click the map',
  },
};

function StopField({ field, stop, icon, inputRef, autoFocus, onFocus, onSelect, onClear }) {
  // null while not editing, so the input shows the chosen stop's label
  const [query, setQuery] = useState(null);
  const [remote, setRemote] = useState({ query: '', results: [] });
  const [highlighted, setHighlighted] = useState(0);
  const [isOpen, setIsOpen] = useState(false);

  const trimmed = query?.trim() ?? '';
  const roadMatches = useMemo(() => searchRoads(trimmed), [trimmed]);
  // Only show place results that belong to what's typed now, not an older query
  const placeMatches = remote.query === trimmed ? remote.results : [];
  const suggestions = [...roadMatches, ...placeMatches];
  const isSearching = trimmed.length >= MIN_PLACE_QUERY_LENGTH && remote.query !== trimmed;
  // Place results arriving can shrink the list under the highlighted row
  const activeIndex = Math.min(highlighted, suggestions.length - 1);

  useEffect(() => {
    if (trimmed.length < MIN_PLACE_QUERY_LENGTH) return undefined;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const results = await searchPlaces(trimmed, controller.signal);
        setRemote({ query: trimmed, results });
      } catch (error) {
        // Place search is a convenience; road matches and map clicks still work
        if (error.name !== 'AbortError') setRemote({ query: trimmed, results: [] });
      }
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [trimmed]);

  function choose(suggestion) {
    onSelect(field, { coordinates: suggestion.coordinates, label: suggestion.label }, { fromSearch: true });
    setQuery(null);
    setIsOpen(false);
  }

  function handleKeyDown(event) {
    if (event.key === 'ArrowDown' && suggestions.length > 0) {
      event.preventDefault();
      setIsOpen(true);
      setHighlighted(() => (activeIndex + 1) % suggestions.length);
    } else if (event.key === 'ArrowUp' && suggestions.length > 0) {
      event.preventDefault();
      setHighlighted((index) => (index - 1 + suggestions.length) % suggestions.length);
    } else if (event.key === 'Enter' && isOpen && suggestions[activeIndex]) {
      event.preventDefault();
      choose(suggestions[activeIndex]);
    } else if (event.key === 'Escape') {
      setQuery(null);
      setIsOpen(false);
    }
  }

  const listId = `${field}-suggestions`;
  const showList = isOpen && trimmed.length >= 2;
  const config = FIELDS[field];

  return (
    <div className={`stop-field stop-field--${field}`}>
      <span className="stop-field__icon" aria-hidden="true">{icon}</span>
      <div className="stop-field__control">
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-label={config.label}
          aria-autocomplete="list"
          aria-expanded={showList}
          aria-controls={listId}
          aria-activedescendant={showList && suggestions[activeIndex] ? `${listId}-${activeIndex}` : undefined}
          autoFocus={autoFocus}
          autoComplete="off"
          spellCheck={false}
          placeholder={config.placeholder}
          value={query ?? stop?.label ?? ''}
          onFocus={(event) => {
            onFocus(field);
            event.target.select();
          }}
          onChange={(event) => {
            setQuery(event.target.value);
            setHighlighted(0);
            setIsOpen(true);
          }}
          onBlur={() => {
            // Typed text that wasn't turned into a place isn't a location, so
            // fall back to showing the current stop
            setQuery(null);
            setIsOpen(false);
          }}
          onKeyDown={handleKeyDown}
          className="stop-field__input"
        />
        {stop && query === null && (
          <button
            type="button"
            className="stop-field__clear"
            aria-label={`Clear ${config.label.toLowerCase()}`}
            onClick={() => onClear(field)}
          >
            ×
          </button>
        )}
      </div>

      {showList && (
        <ul id={listId} role="listbox" aria-label={`${config.label} suggestions`} className="stop-suggestions">
          {suggestions.map((suggestion, index) => (
            <li
              key={suggestion.id}
              id={`${listId}-${index}`}
              role="option"
              aria-selected={index === activeIndex}
              className="stop-suggestions__item"
              title={suggestion.detail ? `${suggestion.label}, ${suggestion.detail}` : suggestion.label}
              // mousedown, not click: click fires after the input's blur has
              // already closed the list
              onMouseDown={(event) => {
                event.preventDefault();
                choose(suggestion);
              }}
              onMouseEnter={() => setHighlighted(index)}
            >
              <span className="stop-suggestions__label">{suggestion.label}</span>
              {suggestion.detail && <span className="stop-suggestions__detail">{suggestion.detail}</span>}
            </li>
          ))}
          {suggestions.length === 0 && (
            <li className="stop-suggestions__empty" role="presentation">
              {isSearching ? 'Searching…' : 'No matches in the study area. Try clicking the map.'}
            </li>
          )}
        </ul>
      )}
    </div>
  );
}

export default function RouteStopsInput({ stops, onFocusStop, onSelectStop, onClearStop, onSwap }) {
  const destinationRef = useRef(null);

  function handleSelect(field, stop, options) {
    onSelectStop(field, stop, options);
    // Like a maps app: after picking the start, move straight on to the destination
    if (field === 'origin' && !stops.destination) destinationRef.current?.focus();
  }

  return (
    <div className="route-stops" role="group" aria-label="Route start and destination">
      <div className="route-stops__fields">
        <StopField
          field="origin"
          stop={stops.origin}
          icon={<span className="stop-field__origin-dot" />}
          autoFocus
          onFocus={onFocusStop}
          onSelect={handleSelect}
          onClear={onClearStop}
        />
        <StopField
          field="destination"
          stop={stops.destination}
          icon={
            <svg viewBox="0 0 24 24">
              <path d="M12 2a7 7 0 0 0-7 7c0 5.2 7 13 7 13s7-7.8 7-13a7 7 0 0 0-7-7zm0 9.5A2.5 2.5 0 1 1 12 6.5a2.5 2.5 0 0 1 0 5z" />
            </svg>
          }
          inputRef={destinationRef}
          onFocus={onFocusStop}
          onSelect={handleSelect}
          onClear={onClearStop}
        />
      </div>

      <button
        type="button"
        className="route-stops__swap"
        onClick={onSwap}
        disabled={!stops.origin && !stops.destination}
        aria-label="Swap starting point and destination"
        title="Swap starting point and destination"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M8 3 4 7h3v7h2V7h3L8 3zm8 18 4-4h-3v-7h-2v7h-3l4 4z" />
        </svg>
      </button>
    </div>
  );
}
