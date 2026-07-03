import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ReportView from '../components/ReportView';
import { ResearchReport } from '../services/api';

const mockReport: ResearchReport = {
  id: 'report-uuid-123',
  ticker: 'AAPL',
  exchange: 'NYSE / NASDAQ',
  generated_at: '2026-07-01T10:00:00Z',
  news: {
    ticker: 'AAPL',
    headlines: [
      {
        title: 'Apple earnings beat expectations',
        url: 'https://example.com',
        published_at: '2026-07-01T09:00:00Z',
        sentiment: 'Positive',
        source: 'Reuters',
      },
    ],
    overall_sentiment: 'Positive',
    key_events: ['Catalyst Earnings beat'],
  },
  financials: {
    ticker: 'AAPL',
    company_name: 'Apple Inc.',
    exchange: 'NYSE / NASDAQ',
    pe_ratio: 30.5,
    eps: 6.2,
    revenue: 380000000000.0,
    market_cap: 3000000000000.0,
    debt_to_equity: 120.0,
    dividend_yield: 0.5,
    fifty_two_week_high: 220.0,
    fifty_two_week_low: 150.0,
    current_price: 180.0,
    valuation_commentary: 'Apple looks fairly valued.',
    data_source: 'yfinance',
  },
  synthesis: {
    verdict: 'BUY',
    confidence: 'High',
    reasoning: 'Strong earnings growth coupled with robust market share makes Apple a buy.',
    risks: ['EU regulatory fine risks', 'Slowing hardware upgrade cycles', 'Foreign exchange headwinds'],
    disclaimer: 'Informational use only.',
  },
  processing_time_seconds: 4.5,
  error: null,
};

describe('ReportView Component', () => {
  it('renders general stock identifiers', () => {
    render(<ReportView report={mockReport} />);

    expect(screen.getAllByText('AAPL').length).toBeGreaterThan(0);
    expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    expect(screen.getByText('NYSE / NASDAQ')).toBeInTheDocument();
  });

  it('renders verdict and confidence details', () => {
    render(<ReportView report={mockReport} />);

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('Confidence: High')).toBeInTheDocument();
    expect(screen.getByText(/Strong earnings growth coupled/)).toBeInTheDocument();
  });

  it('formats large numbers correctly (Market Cap / Revenue)', () => {
    render(<ReportView report={mockReport} />);

    // Market Cap = 3T -> $3.00T
    expect(screen.getByText('$3.00T')).toBeInTheDocument();
    // Revenue = 380B -> $380.00B
    expect(screen.getByText('$380.00B')).toBeInTheDocument();
  });

  it('renders financial metrics lists', () => {
    render(<ReportView report={mockReport} />);

    expect(screen.getByText('30.50')).toBeInTheDocument(); // P/E Ratio
    expect(screen.getByText('6.20')).toBeInTheDocument();  // EPS
    expect(screen.getByText('Apple looks fairly valued.')).toBeInTheDocument(); // Commentary
  });

  it('renders sentiment items and key events', () => {
    render(<ReportView report={mockReport} />);

    expect(screen.getByText('Apple earnings beat expectations')).toBeInTheDocument();
    expect(screen.getAllByText('Positive').length).toBeGreaterThan(0);
    expect(screen.getByText('Catalyst Earnings beat')).toBeInTheDocument();
  });

  it('renders risk factors list items', () => {
    render(<ReportView report={mockReport} />);

    expect(screen.getByText('EU regulatory fine risks')).toBeInTheDocument();
    expect(screen.getByText('Slowing hardware upgrade cycles')).toBeInTheDocument();
  });
});
