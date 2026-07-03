import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import SkeletonLoader from '../components/SkeletonLoader';

describe('SkeletonLoader Component', () => {
  it('renders initial spinner and stage text', () => {
    render(<SkeletonLoader />);

    expect(screen.getByText('Running Multi-Agent Synthesis')).toBeInTheDocument();
    expect(screen.getByText('Initializing research agents...')).toBeInTheDocument();
  });

  it('renders visual card loading guides', () => {
    render(<SkeletonLoader />);

    expect(screen.getByText(/takes between 15 to 30 seconds/i)).toBeInTheDocument();
  });
});
