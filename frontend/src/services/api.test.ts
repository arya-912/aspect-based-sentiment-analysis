/**
 * `services/api.ts` -- every failure normalised into one shape.
 *
 * The distinction the UI depends on is `status: 0` for "never reached the
 * server" versus a real HTTP status. The health banner renders different copy
 * for the two, and it cannot tell them apart if this module gets it wrong: an
 * unreachable backend and a 503 from a loaded-but-degraded one need different
 * advice.
 */

import axios from 'axios'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const post = vi.fn()
const get = vi.fn()

vi.mock('axios', () => {
  const isAxiosError = (error: unknown): boolean =>
    typeof error === 'object' && error !== null && (error as { isAxiosError?: boolean }).isAxiosError === true
  return {
    default: {
      create: () => ({ post, get }),
      isAxiosError,
    },
    isAxiosError,
  }
})

const { ApiRequestError, getHealth, getModelMeta, isApiError, predictSentiment } = await import(
  './api'
)

/** An axios error carrying a response, i.e. the server answered non-2xx. */
function responseError(status: number, data: unknown) {
  return { isAxiosError: true, message: `Request failed with status code ${status}`, response: { status, data } }
}

/** An axios error with no response at all: down, or blocked by CORS. */
function transportError() {
  return { isAxiosError: true, message: 'Network Error', response: undefined }
}

beforeEach(() => {
  post.mockReset()
  get.mockReset()
})

describe('predictSentiment', () => {
  it('posts the text and aspect and returns the body', async () => {
    const body = { aspect: 'screen', sentiment: 'negative', confidence: 0.72 }
    post.mockResolvedValue({ data: body })

    await expect(predictSentiment('The screen is dim.', 'screen')).resolves.toEqual(body)
    expect(post).toHaveBeenCalledWith('/predict', {
      text: 'The screen is dim.',
      aspect: 'screen',
    })
  })

  it('surfaces the backend’s own error message and status', async () => {
    // The 422 the aspect-presence gate returns: its message is the only
    // explanation the user gets, so it must survive the round trip.
    post.mockRejectedValue(
      responseError(422, { error: 'aspect "keyboard" does not occur in the review text', status: 422 }),
    )

    await expect(predictSentiment('The screen is dim.', 'keyboard')).rejects.toMatchObject({
      status: 422,
      message: 'aspect "keyboard" does not occur in the review text',
    })
  })

  it('reports status 0 when the request never reached the server', async () => {
    post.mockRejectedValue(transportError())

    const error = await predictSentiment('x', 'y').catch((e: unknown) => e)
    expect(isApiError(error)).toBe(true)
    expect((error as { status: number }).status).toBe(0)
    expect((error as { message: string }).message).toContain('Is the backend running?')
  })

  it('falls back to a status-shaped message when the body carries none', async () => {
    // A proxy or an upstream crash can return HTML, or nothing.
    post.mockRejectedValue(responseError(502, '<html>Bad Gateway</html>'))

    await expect(predictSentiment('x', 'y')).rejects.toMatchObject({
      status: 502,
      message: '<html>Bad Gateway</html>',
    })
  })

  it.each([
    ['an empty object', {}],
    ['null', null],
    ['an empty string', ''],
    ['a number', 500],
  ])('falls back to the status line for %s', async (_label, data) => {
    post.mockRejectedValue(responseError(500, data))

    await expect(predictSentiment('x', 'y')).rejects.toMatchObject({
      status: 500,
      message: 'Request failed with status 500',
    })
  })

  it('reads FastAPI’s own `detail` key as well as the backend’s `error`', async () => {
    // Every handled error uses {error, status}, but an unhandled 404 from the
    // router can still arrive as {detail}.
    post.mockRejectedValue(responseError(404, { detail: 'Not Found' }))

    await expect(predictSentiment('x', 'y')).rejects.toMatchObject({
      status: 404,
      message: 'Not Found',
    })
  })

  it('wraps a non-axios throw rather than letting it escape untyped', async () => {
    post.mockRejectedValue(new TypeError('something unrelated broke'))

    const error = await predictSentiment('x', 'y').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiRequestError)
    expect(error).toMatchObject({ status: 0, message: 'something unrelated broke' })
  })

  it('wraps a thrown non-Error too', async () => {
    post.mockRejectedValue('just a string')

    await expect(predictSentiment('x', 'y')).rejects.toMatchObject({
      status: 0,
      message: 'An unexpected error occurred.',
    })
  })
})

describe('getModelMeta and getHealth', () => {
  it('return their bodies', async () => {
    get.mockResolvedValue({ data: { status: 'ok', model_loaded: true } })
    await expect(getHealth()).resolves.toEqual({ status: 'ok', model_loaded: true })
  })

  it('pass an abort signal through, so a page can cancel on unmount', async () => {
    get.mockResolvedValue({ data: {} })
    const controller = new AbortController()
    await getModelMeta(controller.signal)
    expect(get).toHaveBeenCalledWith('/model', { signal: controller.signal })
  })

  it.each([
    ['getHealth', () => getHealth()],
    ['getModelMeta', () => getModelMeta()],
  ])('%s normalises a transport failure to status 0', async (_label, call) => {
    get.mockRejectedValue(transportError())
    await expect(call()).rejects.toMatchObject({ status: 0 })
  })

  it('surfaces a 503 from a degraded backend as a real status', async () => {
    // Distinct from status 0: the service is up and saying it is not ready,
    // which is worth retrying rather than a reason to check the process.
    get.mockRejectedValue(responseError(503, { error: 'model not loaded', status: 503 }))
    await expect(getModelMeta()).rejects.toMatchObject({ status: 503, message: 'model not loaded' })
  })
})

describe('isApiError', () => {
  it('accepts the module’s own error', () => {
    expect(isApiError(new ApiRequestError('nope', 503))).toBe(true)
  })

  it('accepts any object of the right shape', () => {
    expect(isApiError({ message: 'nope', status: 500 })).toBe(true)
  })

  it.each([
    ['a plain Error', new Error('nope')],
    ['a string', 'nope'],
    ['null', null],
    ['undefined', undefined],
    ['a partial shape', { message: 'nope' }],
    ['a wrong-typed status', { message: 'nope', status: '500' }],
  ])('rejects %s', (_label, value) => {
    expect(isApiError(value)).toBe(false)
  })
})

describe('ApiRequestError', () => {
  it('is a real Error, so a console trace is usable', () => {
    const error = new ApiRequestError('nope', 422)
    expect(error).toBeInstanceOf(Error)
    expect(error.name).toBe('ApiRequestError')
    expect(error.status).toBe(422)
    expect(error.message).toBe('nope')
  })
})

describe('the axios mock itself', () => {
  it('is in place, so these tests make no real request', () => {
    expect(vi.isMockFunction(post)).toBe(true)
    expect(axios.create).toBeDefined()
  })
})
