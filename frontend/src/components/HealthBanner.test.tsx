/**
 * `HealthBanner` -- telling apart two failures that look identical from the UI.
 *
 * An unreachable API and an API that is up with no model loaded need different
 * advice, and the backend goes out of its way to make them distinguishable:
 * it answers 200 `{"status": "degraded"}` for the second rather than 503.
 * This component is the only place that distinction is acted on.
 *
 * The polling rule is the other thing worth pinning: re-poll every 5s while
 * unhealthy, stop for good on the first success, so a working app issues
 * exactly one request.
 */

import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import HealthBanner from './HealthBanner'
import { ApiRequestError, getHealth } from '../services/api'

vi.mock('../services/api', async () => {
  const actual = await vi.importActual<typeof import('../services/api')>('../services/api')
  return {
    ...actual,
    BASE_URL: 'http://localhost:8000',
    getHealth: vi.fn(),
  }
})

const mockedGetHealth = vi.mocked(getHealth)

beforeEach(() => {
  mockedGetHealth.mockReset()
})

afterEach(() => {
  vi.useRealTimers()
})

describe('the healthy case', () => {
  it('renders nothing at all', async () => {
    mockedGetHealth.mockResolvedValue({ status: 'ok', model_loaded: true })
    const { container } = render(<HealthBanner />)
    await waitFor(() => expect(mockedGetHealth).toHaveBeenCalled())
    expect(container).toBeEmptyDOMElement()
  })

  it('never flashes a banner while the first check is in flight', () => {
    // `checking` renders nothing, so a healthy load is visually silent.
    mockedGetHealth.mockReturnValue(new Promise(() => {}))
    const { container } = render(<HealthBanner />)
    expect(container).toBeEmptyDOMElement()
  })

  it('issues exactly one request and then settles', async () => {
    vi.useFakeTimers()
    mockedGetHealth.mockResolvedValue({ status: 'ok', model_loaded: true })
    render(<HealthBanner />)
    await vi.waitFor(() => expect(mockedGetHealth).toHaveBeenCalledTimes(1))

    await vi.advanceTimersByTimeAsync(30_000)
    expect(mockedGetHealth).toHaveBeenCalledTimes(1)
  })
})

describe('up, but no model loaded', () => {
  beforeEach(() => {
    mockedGetHealth.mockResolvedValue({ status: 'degraded', model_loaded: false })
  })

  it('says the server is running and the model is not', async () => {
    render(<HealthBanner />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('No model is loaded')
    expect(alert).toHaveTextContent('started without an artifact')
  })

  it('advises training one, not starting the server', async () => {
    render(<HealthBanner />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('python train.py train')
    expect(alert).not.toHaveTextContent('uvicorn')
  })
})

describe('unreachable', () => {
  beforeEach(() => {
    mockedGetHealth.mockRejectedValue(
      new ApiRequestError('Could not reach the API at http://localhost:8000.', 0),
    )
  })

  it('says it cannot reach the API', async () => {
    render(<HealthBanner />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Cannot reach the API')
  })

  it('advises starting the server, not training a model', async () => {
    render(<HealthBanner />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('uvicorn app.main:app --port 8000')
    expect(alert).not.toHaveTextContent('python train.py train')
  })

  it('shows the error’s own message when it has one', async () => {
    render(<HealthBanner />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Could not reach the API at http://localhost:8000.')
  })

  it('falls back to naming the base URL for an unrecognisable failure', async () => {
    mockedGetHealth.mockRejectedValue('something odd')
    render(<HealthBanner />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('No response from http://localhost:8000.')
  })
})

describe('polling', () => {
  it('re-checks every 5s while unhealthy', async () => {
    vi.useFakeTimers()
    mockedGetHealth.mockRejectedValue(new ApiRequestError('down', 0))
    render(<HealthBanner />)

    await vi.waitFor(() => expect(mockedGetHealth).toHaveBeenCalledTimes(1))
    await vi.advanceTimersByTimeAsync(5000)
    expect(mockedGetHealth).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(5000)
    expect(mockedGetHealth).toHaveBeenCalledTimes(3)
  })

  it('clears the banner once the backend comes up, with no reload', async () => {
    vi.useFakeTimers()
    mockedGetHealth
      .mockRejectedValueOnce(new ApiRequestError('down', 0))
      .mockResolvedValue({ status: 'ok', model_loaded: true })

    const { container } = render(<HealthBanner />)
    await vi.waitFor(() => expect(screen.queryByRole('alert')).not.toBeNull())

    await vi.advanceTimersByTimeAsync(5000)
    await vi.waitFor(() => expect(container).toBeEmptyDOMElement())
  })

  it('stops polling after unmount', async () => {
    vi.useFakeTimers()
    mockedGetHealth.mockRejectedValue(new ApiRequestError('down', 0))
    const { unmount } = render(<HealthBanner />)
    await vi.waitFor(() => expect(mockedGetHealth).toHaveBeenCalledTimes(1))

    unmount()
    await vi.advanceTimersByTimeAsync(20_000)
    // A component that kept polling after unmount would also be setting state
    // on an unmounted tree.
    expect(mockedGetHealth).toHaveBeenCalledTimes(1)
  })

  it('states the retry interval in the banner', async () => {
    mockedGetHealth.mockRejectedValue(new ApiRequestError('down', 0))
    render(<HealthBanner />)
    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('Rechecking every 5s')
  })
})
