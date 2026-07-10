"""Tests for SupabaseClientPool connection reuse."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from api.auth import SupabaseClientPool


def test_pool_reuses_clients():
    """Verify clients are reused from the pool instead of creating new ones."""
    async def run():
        pool = SupabaseClientPool(pool_size=3, supabase_url="http://test", anon_key="test-key")

        with patch("api.auth.acreate_client", new_callable=AsyncMock) as mock_create:
            mock_client = AsyncMock()
            mock_client.postgrest = MagicMock()
            mock_create.return_value = mock_client

            # First acquisition: should create a new client
            client1 = await pool.acquire()
            assert mock_create.call_count == 1, "Should create first client"

            # Release it back to the pool
            await pool.release(client1)
            assert len(pool._idle_clients) == 1, "Client should be in idle pool"

            # Second acquisition: should reuse the one we just released
            client2 = await pool.acquire()
            assert mock_create.call_count == 1, "Should NOT create a new client (reused)"
            assert client2 is client1, "Should be the same client object"
            assert len(pool._idle_clients) == 0, "No idle clients now"

    asyncio.run(run())


def test_pool_backpressure_at_capacity():
    """Verify pool applies backpressure when all slots are in use."""
    async def run():
        pool = SupabaseClientPool(pool_size=2, supabase_url="http://test", anon_key="test-key")

        with patch("api.auth.acreate_client", new_callable=AsyncMock) as mock_create:
            mock_client = AsyncMock()
            mock_client.postgrest = MagicMock()
            mock_create.return_value = mock_client

            # Acquire 2 clients (fill the pool)
            client1 = await pool.acquire()
            client2 = await pool.acquire()
            assert mock_create.call_count == 2

            # Start a third acquisition (should block)
            third_acquire_task = asyncio.create_task(pool.acquire())
            await asyncio.sleep(0.1)  # Let it start waiting
            assert not third_acquire_task.done(), "Should be waiting for a slot"

            # Release one → third acquisition should proceed
            await pool.release(client1)
            await asyncio.wait_for(third_acquire_task, timeout=1.0)
            assert mock_create.call_count == 2, "Should still be 2 (reused client1)"

    asyncio.run(run())


def test_pool_shutdown_closes_clients():
    """Verify all clients are properly closed on shutdown."""
    async def run():
        pool = SupabaseClientPool(pool_size=3, supabase_url="http://test", anon_key="test-key")

        with patch("api.auth.acreate_client", new_callable=AsyncMock) as mock_create:
            # Create two distinct mock clients
            mock_client1 = AsyncMock()
            mock_client1.postgrest.aclose = AsyncMock()

            mock_client2 = AsyncMock()
            mock_client2.postgrest.aclose = AsyncMock()

            # First call returns client1, second returns client2 (but only if we exhaust client1)
            mock_create.side_effect = [mock_client1, mock_client2]

            # Acquire first client, hold it (don't release yet)
            client1 = await pool.acquire()

            # Now acquire a second one (pool is at capacity 1, so this creates a new one)
            client2 = await pool.acquire()

            # Release both
            await pool.release(client1)
            await pool.release(client2)

            assert len(pool._idle_clients) == 2, f"Expected 2 idle, got {len(pool._idle_clients)}"

            # Shutdown should close all idle clients
            await pool.shutdown()
            assert len(pool._idle_clients) == 0
            assert mock_client1.postgrest.aclose.call_count == 1, "Should close client1"
            assert mock_client2.postgrest.aclose.call_count == 1, "Should close client2"

    asyncio.run(run())


def test_pool_handles_close_errors_gracefully():
    """Verify shutdown doesn't crash if closing a client fails."""
    async def run():
        pool = SupabaseClientPool(pool_size=2, supabase_url="http://test", anon_key="test-key")

        with patch("api.auth.acreate_client", new_callable=AsyncMock) as mock_create:
            # First client closes fine, second raises an error
            mock_client1 = AsyncMock()
            mock_client1.postgrest.aclose = AsyncMock()

            mock_client2 = AsyncMock()
            mock_client2.postgrest.aclose = AsyncMock(side_effect=RuntimeError("Close failed"))

            mock_create.side_effect = [mock_client1, mock_client2]

            client1 = await pool.acquire()
            client2 = await pool.acquire()
            await pool.release(client1)
            await pool.release(client2)

            # Shutdown should handle the error and continue
            await pool.shutdown()  # Should not raise
            assert len(pool._idle_clients) == 0

    asyncio.run(run())


def test_pool_survives_create_failure():
    """A failure inside acreate_client must release the semaphore slot.

    Otherwise a transient outage during client creation permanently shrinks the
    pool, and enough failures deadlock it. Regression test for that bug.
    """
    async def run():
        pool = SupabaseClientPool(pool_size=1, supabase_url="http://bad", anon_key="x")

        with patch("api.auth.acreate_client", new_callable=AsyncMock) as mock_create:
            # First create blows up; second succeeds.
            good_client = AsyncMock()
            good_client.postgrest = MagicMock()
            mock_create.side_effect = [RuntimeError("connection failed"), good_client]

            # First acquire fails.
            try:
                await pool.acquire()
                assert False, "expected the create to raise"
            except RuntimeError:
                pass

            # The slot must have been returned: a second acquire must not hang.
            client = await asyncio.wait_for(pool.acquire(), timeout=1.0)
            assert client is good_client
            await pool.release(client)

    asyncio.run(run())


def test_concurrent_acquisitions():
    """Stress test: multiple concurrent requests competing for pool clients."""
    async def run():
        pool = SupabaseClientPool(pool_size=5, supabase_url="http://test", anon_key="test-key")

        with patch("api.auth.acreate_client", new_callable=AsyncMock) as mock_create:
            mock_client = AsyncMock()
            mock_client.postgrest = MagicMock()
            mock_create.return_value = mock_client

            async def use_client():
                """Acquire, hold briefly, release."""
                client = await pool.acquire()
                await asyncio.sleep(0.01)
                await pool.release(client)
                return mock_create.call_count

            # Run 20 concurrent "requests" against a pool of 5
            results = await asyncio.gather(*[use_client() for _ in range(20)])

            # Should have created only 5 clients total, not 20
            assert mock_create.call_count == 5, f"Created {mock_create.call_count}, expected 5"
            assert all(r == 5 for r in results), "Final count should always be 5"

    asyncio.run(run())
