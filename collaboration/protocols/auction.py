"""
Multi-Auction Protocol System - Supports English, Dutch & Sealed-Bid Auctions
"""

from __future__ import annotations
import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

# Core dependencies
from core.agent.base import AgentBase
from kafka.producer import KafkaProducer
from utils.logger import get_logger
from utils.metrics import ProtocolMetrics
from utils.serialization import serialize

logger = get_logger(__name__)
metrics = ProtocolMetrics()

class AuctionType(Enum):
    ENGLISH = auto()
    DUTCH = auto()
    SEALED_BID = auto()
    VICKREY = auto()

class AuctionState(Enum):
    INITIATED = auto()
    RUNNING = auto()
    PAUSED = auto()
    CLOSED = auto()
    FAILED = auto()

class BidValidity(Enum):
    VALID = auto()
    INSUFFICIENT_BID = auto()
    LATE_BID = auto()
    INVALID_BIDDER = auto()

@dataclass(frozen=True)
class Auction:
    auction_id: str
    item: Dict[str, Any]
    auction_type: AuctionType
    reserve_price: float
    created_at: datetime
    creator_id: str
    terms: Dict[str, Any]
    end_time: datetime
    bid_increment: float = 0.0
    current_price: float = 0.0

@dataclass(frozen=True)
class Bid:
    bid_id: str
    auction_id: str
    bidder_id: str
    amount: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class AuctionProtocol:
    """
    Distributed Auction Management System with:
    - Multi-round bidding
    - Dynamic price adjustments
    - Anti-sniping protection
    - Fraud detection
    """
    
    def __init__(
        self,
        agent: AgentBase,
        producer: KafkaProducer,
        bid_timeout: int = 60,
        anti_sniping_window: int = 5
    ):
        self.agent = agent
        self.producer = producer
        self.active_auctions: Dict[str, Auction] = {}
        self.bid_registry: Dict[str, List[Bid]] = {}
        self.bid_timeout = bid_timeout
        self.anti_sniping_window = anti_sniping_window
        self._lock = asyncio.Lock()
        self._auction_tasks: Dict[str, asyncio.Task] = {}

    async def create_auction(
        self,
        item: Dict[str, Any],
        auction_type: AuctionType,
        reserve_price: float,
        duration: int,
        terms: Dict[str, Any]
    ) -> str:
        """Initialize new auction"""
        auction_id = f"auction_{uuid.uuid4().hex}"
        now = datetime.utcnow()
        
        auction = Auction(
            auction_id=auction_id,
            item=item,
            auction_type=auction_type,
            reserve_price=reserve_price,
            created_at=now,
            creator_id=self.agent.agent_id,
            terms=terms,
            end_time=now + timedelta(seconds=duration),
            current_price=reserve_price if auction_type == AuctionType.DUTCH else 0.0
        )
        
        async with self._lock:
            self.active_auctions[auction_id] = auction
            self.bid_registry[auction_id] = []
            
        await self.producer.send(
            topic="auction_events",
            key=auction_id,
            value=serialize(auction),
            headers={
                "protocol": "auction/v2",
                "type": auction_type.name
            }
        )
        
        self._auction_tasks[auction_id] = asyncio.create_task(
            self._manage_auction_timeline(auction_id)
        )
        return auction_id

    async def submit_bid(self, auction_id: str, bidder_id: str, amount: float) -> BidValidity:
        """Process bid submission with validity checks"""
        async with self._lock:
            if auction_id not in self.active_auctions:
                return BidValidity.LATE_BID
                
            auction = self.active_auctions[auction_id]
            current_bids = self.bid_registry[auction_id]
            
            validity = self._validate_bid(auction, bidder_id, amount)
            if validity != BidValidity.VALID:
                return validity
                
            bid = Bid(
                bid_id=f"bid_{uuid.uuid4().hex}",
                auction_id=auction_id,
                bidder_id=bidder_id,
                amount=amount,
                timestamp=datetime.utcnow()
            )
            
            current_bids.append(bid)
            
            # Update auction state
            if auction.auction_type == AuctionType.ENGLISH:
                self.active_auctions[auction_id] = replace(
                    auction,
                    current_price=amount + auction.bid_increment
                )
                
            await self._handle_bid_extension(auction_id)
            
        await self.producer.send(
            topic=f"{bidder_id}_bids",
            key=bid.bid_id,
            value=serialize(bid)
        )
        metrics.bids_processed.labels(
            auction_type=auction.auction_type.name
        ).inc()
        return BidValidity.VALID

    def _validate_bid(self, auction: Auction, bidder_id: str, amount: float) -> BidValidity:
        """Validate bid against auction rules"""
        if datetime.utcnow() > auction.end_time:
            return BidValidity.LATE_BID
            
        if bidder_id == auction.creator_id:
            return BidValidity.INVALID_BIDDER
            
        if auction.auction_type == AuctionType.ENGLISH:
            if amount < (auction.current_price + auction.bid_increment):
                return BidValidity.INSUFFICIENT_BID
        elif auction.auction_type == AuctionType.DUTCH:
            if amount > auction.current_price:
                return BidValidity.INSUFFICIENT_BID
                
        return BidValidity.VALID

    async def _manage_auction_timeline(self, auction_id: str):
        """Handle auction lifecycle including anti-sniping extensions"""
        try:
            while True:
                now = datetime.utcnow()
                async with self._lock:
                    auction = self.active_auctions.get(auction_id)
                    if not auction or now > auction.end_time:
                        break
                        
                    remaining = (auction.end_time - now).total_seconds()
                    await asyncio.sleep(min(remaining, 1.0))
                    
                # Check for anti-sniping window
                if remaining <= self.anti_sniping_window:
                    async with self._lock:
                        auction = self.active_auctions[auction_id]
                        if any(bid.timestamp > auction.end_time - timedelta(seconds=self.anti_sniping_window)
                               for bid in self.bid_registry[auction_id]):
                            new_end_time = auction.end_time + timedelta(seconds=self.anti_sniping_window)
                            self.active_auctions[auction_id] = replace(
                                auction,
                                end_time=new_end_time
                            )
                            logger.info(f"Extended auction {auction_id} due to sniping")
                            
            await self._finalize_auction(auction_id)
            
        except asyncio.CancelledError:
            logger.warning(f"Auction {auction_id} monitoring cancelled")

    async def _finalize_auction(self, auction_id: str):
        """Determine auction outcome and notify participants"""
        async with self._lock:
            auction = self.active_auctions.pop(auction_id, None)
            bids = self.bid_registry.pop(auction_id, [])
            
        if not bids:
            logger.info(f"Auction {auction_id} closed with no bids")
            metrics.auctions_failed.inc()
            return
            
        winner = self._determine_winner(auction, bids)
        await self._notify_parties(auction, winner, bids)
        metrics.auctions_completed.inc()

    def _determine_winner(self, auction: Auction, bids: List[Bid]) -> Optional[Bid]:
        """Select winning bid based on auction rules"""
        if auction.auction_type == AuctionType.ENGLISH:
            return max(bids, key=lambda b: b.amount)
        elif auction.auction_type == AuctionType.DUTCH:
            return next((b for b in bids if b.amount <= auction.current_price), None)
        elif auction.auction_type == AuctionType.VICKREY:
            sorted_bids = sorted(bids, key=lambda b: -b.amount)
            return sorted_bids[1] if len(sorted_bids) > 1 else None
        return None

    async def _notify_parties(self, auction: Auction, winner: Optional[Bid], all_bids: List[Bid]):
        """Send notifications to winner and participants"""
        if winner:
            await self.producer.send(
                topic=f"{winner.bidder_id}_wins",
                key=auction.auction_id,
                value=serialize({
                    "auction_id": auction.auction_id,
                    "amount": winner.amount,
                    "item": auction.item
                })
            )
            
        for bid in all_bids:
            if bid != winner:
                await self.producer.send(
                    topic=f"{bid.bidder_id}_results",
                    key=auction.auction_id,
                    value=serialize({
                        "auction_id": auction.auction_id,
                        "status": "lost",
                        "winning_bid": winner.amount if winner else None
                    })
                )

    async def _handle_bid_extension(self, auction_id: str):
        """Extend auction deadline on new bids (anti-sniping)"""
        async with self._lock:
            auction = self.active_auctions[auction_id]
            new_end_time = datetime.utcnow() + timedelta(seconds=self.bid_timeout)
            if new_end_time > auction.end_time:
                self.active_auctions[auction_id] = replace(
                    auction,
                    end_time=new_end_time
                )

class AuctionAgent(AgentBase):
    """Agent with integrated auction capabilities"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.producer = KafkaProducer()
        self.auction_protocol = AuctionProtocol(
            agent=self,
            producer=self.producer
        )
        
    async def handle_auction_event(self, auction: Auction):
        """Sample bid strategy"""
        if auction.auction_type == AuctionType.ENGLISH:
            bid_amount = auction.current_price + auction.bid_increment * 1.1
            await self.auction_protocol.submit_bid(
                auction.auction_id,
                self.agent_id,
                bid_amount
            )

# -------------------------------------------------------------------
# Test Scenario
# -------------------------------------------------------------------

async def test_english_auction():
    seller = AuctionAgent("seller-001")
    buyer1 = AuctionAgent("buyer-001")
    buyer2 = AuctionAgent("buyer-002")
    
    auction_id = await seller.auction_protocol.create_auction(
        item={"type": "gpu", "model": "A100"},
        auction_type=AuctionType.ENGLISH,
        reserve_price=1000.0,
        duration=60,
        terms={"currency": "USD"}
    )
    
    await asyncio.gather(
        buyer1.handle_auction_event(seller.auction_protocol.active_auctions[auction_id]),
        buyer2.handle_auction_event(seller.auction_protocol.active_auctions[auction_id])
    )
    
    # Let auction conclude
    await asyncio.sleep(65)
    
if __name__ == "__main__":
    asyncio.run(test_english_auction())
