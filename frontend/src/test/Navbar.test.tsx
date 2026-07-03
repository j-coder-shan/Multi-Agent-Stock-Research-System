import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import Navbar from '../components/Navbar';

describe('Navbar Component', () => {
  it('renders brand name and tagline', () => {
    const setActiveTab = vi.fn();
    render(<Navbar activeTab="research" setActiveTab={setActiveTab} />);

    expect(screen.getByText('ANTIGRAVITY')).toBeInTheDocument();
    expect(screen.getByText('Multi-Agent Stock Analyst')).toBeInTheDocument();
  });

  it('renders all tab buttons', () => {
    const setActiveTab = vi.fn();
    render(<Navbar activeTab="research" setActiveTab={setActiveTab} />);

    expect(screen.getByText(/Research/i)).toBeInTheDocument();
    expect(screen.getByText(/Report History/i)).toBeInTheDocument();
    expect(screen.getByText(/CSE Explorer/i)).toBeInTheDocument();
  });

  it('applies correct active styling to the active tab', () => {
    const setActiveTab = vi.fn();
    render(<Navbar activeTab="history" setActiveTab={setActiveTab} />);

    const historyBtn = screen.getByText(/Report History/i);
    expect(historyBtn).toHaveClass('bg-indigo-600');
  });

  it('calls setActiveTab when buttons are clicked', () => {
    const setActiveTab = vi.fn();
    render(<Navbar activeTab="research" setActiveTab={setActiveTab} />);

    const cseBtn = screen.getByText(/CSE Explorer/i);
    fireEvent.click(cseBtn);

    expect(setActiveTab).toHaveBeenCalledWith('cse');
  });
});
