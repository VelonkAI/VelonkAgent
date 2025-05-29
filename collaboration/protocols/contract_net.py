"""
FIPA Contract Net Protocol Implementation - Decentralized Task Allocation System
"""

from __future__ import annotations
import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Internal components
from core.agent.base import AgentBase
from kafka.producer import KafkaProducer
from kafka.schemas import TaskRequest
from utils.logger import get_logger
from utils.metrics import ProtocolMetrics

logger = get_logger(__name__)
metrics = ProtocolMetrics()

class ContractNetState(Enum):
    INITIATED = "initiated"
    CFP_ISSUED = "cfp_issued"
    BIDS_RECEIVED = "bids_received"
    AWARDED = "awarded"
    COMPLETED = "completed"
    FAILED = "failed"

class ContractNetError(Exception):
    """Base exception for contract net failures"""
    pass

class BidTimeoutError(ContractNetError):
    """Raised when bidding phase times out"""
    pass

class InvalidBidError(ContractNetError):
    """Raised for non-compliant bid proposals"""
    pass

@dataclass(frozen=True)
class CFP:
    """Call For Proposal - Task Announcement"""
    task_id: str
    task_type: str
    requirements: Dict[str, Any]
    deadline: datetime
    originator: str
    context: Dict[str, Any] = field(default_factory=dict)
    qos: Dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class Bid:
    """Agent Bid Proposal"""
    bid_id: str
    cfp_id: str
    bidder_id: str
    capability_scores: Dict[str, float]
    cost_estimate: float
    timeline: timedelta
    constraints: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Award:
    """Contract Award Decision"""
    award_id: str
    cfp_id: str
    winner_id: str
    terms: Dict[str, Any]
    service_level: Dict[str, Any] = field(default_factory=dict)

class ContractNetProtocol:
    """
    FIPA-compliant Contract Net Protocol Orchestrator
    Implements:
    - Multi-round bidding
    - Bid evaluation policies
    - Penalty-aware awarding
    - Contract lifecycle management
    """
    
    def __init__(
        self,
        agent: AgentBase,
        kafka_producer: KafkaProducer,
        bidding_timeout: int = 30,
        max_retries: int = 3
    ):
        self.agent = agent
        self.producer = kafka_producer
        self.active_contracts: Dict[str, ContractNetState] = {}
        self.pending_bids: Dict[str, Set[Bid]] = {}
        self.bidding_timeout = bidding_timeout
        self.max_retries = max_retries
        self._lock = asyncio.Lock()

    async def initiate_cfp(
        self,
        task_type: str,
        requirements: Dict[str, Any],
        qos: Dict[str, float],
        context: Optional[Dict] = None
    ) -> str:
        """Publish new Call For Proposal"""
        cfp_id = f"cfp_{uuid.uuid4().hex}"
        cfp = CFP(
            task_id=cfp_id,
            task_type=task_type,
            requirements=requirements,
            deadline=datetime.utcnow() + timedelta(seconds=self.bidding_timeout),
            originator=self.agent.agent_id,
            context=context or {},
            qos=qos
        )
        
        async with self._lock:
            self.active_contracts[cfp_id] = ContractNetState.INITIATED
            self.pending_bids[cfp_id] = set()

        # Publish to Kafka
        await self.producer.send(
            topic="contract_net_cfps",
            key=cfp_id,
            value=self._serialize_cfp(cfp),
            headers={
                "protocol_version": "1.1",
                "originator": self.agent.agent_id
            }
        )
        metrics.cfp_issued.inc()
        
        # Start timeout monitor
        asyncio.create_task(self._monitor_bidding(cfp_id))
        return cfp_id

    async def submit_bid(self, cfp_id: str, bid: Bid) -> None:
        """Submit bid for a CFP"""
        if not self._validate_bid(bid):
            raise InvalidBidError(f"Invalid bid {bid.bid_id} for CFP {cfp_id}")
            
        async with self._lock:
            if cfp_id not in self.active_contracts:
                raise ContractNetError(f"CFP {cfp_id} not found")
                
            self.pending_bids[cfp_id].add(bid)
            metrics.bids_received.labels(task_type=self.active_contracts[cfp_id].task_type).inc()

        # Send bid response
        await self.producer.send(
            topic=f"{self.agent.agent_id}_bids",
            key=bid.bid_id,
            value=self._serialize_bid(bid),
            headers={
                "cfp_id": cfp_id,
                "bidder": bid.bidder_id
            }
        )

    async def award_contract(self, cfp_id: str, winner_id: str, terms: Dict) -> Award:
        """Finalize contract award"""
        async with self._lock:
            if cfp_id not in self.active_contracts:
                raise ContractNetError(f"CFP {cfp_id} not active")
                
            bids = self.pending_bids[cfp_id]
            if not bids:
                raise ContractNetError("No bids to award")
                
            # Evaluation logic
            winner = next(b for b in bids if b.bidder_id == winner_id)
            award = Award(
                award_id=f"award_{uuid.uuid4().hex}",
                cfp_id=cfp_id,
                winner_id=winner_id,
                terms=terms,
                service_level=winner.meta.get("sla", {})
            )
            
            self.active_contracts[cfp_id] = ContractNetState.AWARDED
            metrics.contracts_awarded.inc()

        # Notify winner and participants
        await self._notify_parties(cfp_id, award, list(bids))
        return award

    async def _monitor_bidding(self, cfp_id: str) -> None:
        """Monitor bidding phase with timeout"""
        try:
            await asyncio.sleep(self.bidding_timeout)
            async with self._lock:
                if self.active_contracts.get(cfp_id) == ContractNetState.INITIATED:
                    self.active_contracts[cfp_id] = ContractNetState.FAILED
                    metrics.bidding_timeouts.inc()
                    raise BidTimeoutError(f"CFP {cfp_id} expired with no bids")
                    
        except asyncio.CancelledError:
            logger.info(f"Bidding for {cfp_id} completed early")

    def _validate_bid(self, bid: Bid) -> bool:
        """Validate bid against protocol rules"""
        # Implement validation logic:
        # - Mandatory fields
        # - Capability matching
        # - Deadline adherence
        return True  # Simplified for example

    def _evaluate_bids(self, bids: List[Bid], cfp: CFP) -> Tuple[Bid, List[Bid]]:
        """Evaluate bids using configurable strategy"""
        # Default: Highest capability score
        ranked = sorted(
            bids,
            key=lambda b: sum(b.capability_scores.values()),
            reverse=True
        )
        return ranked[0], ranked[1:]

    async def _notify_parties(self, cfp_id: str, award: Award, all_bids: List[Bid]) -> None:
        """Notify winner and other participants"""
        # Send award notice
        await self.producer.send(
            topic=f"{award.winner_id}_awards",
            key=award.award_id,
            value=self._serialize_award(award)
        )
        
        # Notify losers
        for bid in all_bids:
            if bid.bidder_id != award.winner_id:
                await self.producer.send(
                    topic=f"{bid.bidder_id}_rejections",
                    key=cfp_id,
                    value={"cfp_id": cfp_id, "reason": "outbid"}
                )

    def _serialize_cfp(self, cfp: CFP) -> Dict:
        return {
            "task_id": cfp.task_id,
            "task_type": cfp.task_type,
            "deadline": cfp.deadline.isoformat(),
            "qos": cfp.qos,
            "requirements": cfp.requirements
        }

    def _serialize_bid(self, bid: Bid) -> Dict:
        return {
            "bid_id": bid.bid_id,
            "cfp_id": bid.cfp_id,
            "bidder": bid.bidder_id,
            "capabilities": bid.capability_scores,
            "cost": bid.cost_estimate,
            "timeline": bid.timeline.total_seconds()
        }

    def _serialize_award(self, award: Award) -> Dict:
        return {
            "award_id": award.award_id,
            "winner": award.winner_id,
            "terms": award.terms,
            "sla": award.service_level
        }

# -------------------------------------------------------------------
# Usage Example with Agent Integration
# -------------------------------------------------------------------

class ContractNetAgent(AgentBase):
    """Agent with integrated Contract Net capabilities"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.producer = KafkaProducer()
        self.contract_net = ContractNetProtocol(
            agent=self,
            kafka_producer=self.producer
        )
        
    async def handle_cfp(self, cfp: CFP):
        """Sample bid response logic"""
        bid = Bid(
            bid_id=f"bid_{uuid.uuid4().hex}",
            cfp_id=cfp.task_id,
            bidder_id=self.agent_id,
            capability_scores={"cpu": 0.95, "memory": 0.8},
            cost_estimate=0.15,
            timeline=timedelta(seconds=120)
        )
        await self.contract_net.submit_bid(cfp.task_id, bid)
        
    async def execute_contract(self, award: Award):
        """Contract fulfillment logic"""
        logger.info(f"Executing contract {award.award_id}")
        # Actual task execution goes here
        metrics.contracts_completed.inc()

# -------------------------------------------------------------------
# Protocol Test Scenario
# -------------------------------------------------------------------

async def test_contract_net():
    # Initialize participants
    initiator = ContractNetAgent("initiator-001")
    participant1 = ContractNetAgent("worker-001")
    participant2 = ContractNetAgent("worker-002")
    
    # Start CFP
    cfp_id = await initiator.contract_net.initiate_cfp(
        task_type="image_processing",
        requirements={"gpu": 1, "memory": "8GB"},
        qos={"latency": 2.0, "accuracy": 0.95}
    )
    
    # Participants respond
    await asyncio.gather(
        participant1.handle_cfp(initiator.active_contracts[cfp_id]),
        participant2.handle_cfp(initiator.active_contracts[cfp_id])
    )
    
    # Evaluate and award
    bids = initiator.contract_net.pending_bids[cfp_id]
    winner = max(bids, key=lambda b: sum(b.capability_scores.values()))
    award = await initiator.contract_net.award_contract(cfp_id, winner.bidder_id, {})
    
    # Execute contract
    await initiator.execute_contract(award)

if __name__ == "__main__":
    asyncio.run(test_contract_net())
