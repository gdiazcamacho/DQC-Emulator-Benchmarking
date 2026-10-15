from abc import abstractmethod
from typing import List

import numpy as np

from netqasm.sdk.connection import BaseNetQASMConnection
from netqasm.sdk.qubit import Qubit
from squidasm.util.util import get_qubit_state

from squidasm.sim.stack.common import LogManager
from squidasm.sim.stack.program import Program, ProgramContext, ProgramMeta


def CP(control: Qubit, target: Qubit, angle: float = np.pi):
    """Controlled-PHASE gate, CP(a) = diag(1, 1, 1, e^{i a}).

    Despite the historical name `CRz` (kept below as an alias), this sequence
    implements controlled-PHASE, not controlled-Rz. The leading
    `control.rot_Z(angle/2)` is exactly the control-local correction that turns
    CRz into CP:

        CP(a) = CRz(a) . P_control(a/2)

    This matters because CP is the gate the inverse QFT is actually built from,
    and because the `angle < 0` branch below is only valid for CP: CP is
    2*pi-periodic in the angle, whereas Rz is 4*pi-periodic, so the same
    wraparound applied to a true CRz would flip its sign.

    Verified against exact unitaries in tests/test_primitives.py.

    Parameters:
    - control: qubit
    - target: qubit
    - angle: rotation angle in radians
    """

    if angle < 0:
        angle = 2*np.pi + angle

    control.rot_Z(angle=angle/2)
    control.cnot(target)
    target.rot_Z(angle=2*np.pi-angle/2)
    # target.rot_Z(angle=-angle/2)
    control.cnot(target)
    target.rot_Z(angle=angle/2)


# Backwards-compatible alias. The name is a misnomer (see CP above); prefer CP
# in new code.
CRz = CP


def SWAP(qubit_1: Qubit, qubit_2: Qubit):
    qubit_1.cnot(qubit_2)
    qubit_2.cnot(qubit_1)
    qubit_1.cnot(qubit_2)


class AbstractNode(Program):
    """ Abstract class that includes a register of qubits and peers the node is connected to

    Parameters:
    - name: str -> The name of the node
    - peers: List[str] -> The nodes that are connected to this node
    - qubits: int -> The maximum number of qubits that this node can hold
    """
    def __init__(self, *args, name: str, peers: List[str], qubits: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        self._peers = peers
        self._qubits = [None for i in range(qubits)]
        self._max_qubits = qubits
        self.logger = LogManager.get_stack_logger(self.__class__.__name__)

    @property
    def meta(self) -> ProgramMeta:
        return ProgramMeta(
            name=self._name,
            csockets=self._peers,
            epr_sockets=self._peers,
            max_qubits=self._max_qubits,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def peers(self) -> List[str]:
        return self._peers

    @property
    def max_qubits(self) -> int:
        return self._max_qubits

    @property
    def qubits(self) -> List[Qubit]:
        return self._qubits

    @property
    def available_qubits(self):
        """ Indices of qubits that are not used """
        return [i for i, q in enumerate(self.qubits) if q is None]

    @property
    def connection(self):
        return self._connection

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, context: ProgramContext):
        if isinstance(context, ProgramContext):
            self._context = context
            self._connection = self._context.connection

    @property
    def csockets(self):
        return self.context.csockets

    @property
    def epr_sockets(self):
        return self.context.epr_sockets

    @abstractmethod
    def run(self, context: ProgramContext):
        raise NotImplementedError(str(self.__class__) + " must be extended to run program")

    def flush(self):
        return self.connection.flush()

    def init_qubit(self, index=None, *args, **kwargs):
        """ Initialize a new qubit at index with state |0> """
        self.qubits[index] = Qubit(self.connection, *args, **kwargs)

    def free_all_qubits(self):
        for q in self.qubits:
            try:
                q.free()
                q = None
            except:
                pass

    def get_qubit_state(self, qubit: Qubit, *args, **kwargs):
        """ Retrieves the underlying quantum state of qubit index in density matrix formalism """
        if not isinstance(qubit, Qubit):
            raise TypeError(f"{qubit} not Qubit")
        return get_qubit_state(qubit, self.name, **kwargs)

    def generate_epr(self, peer: str, role: str = "send", **kwargs):
        """ Wrapper for epr_socket.create_keep and epr_socket.recv_keep """

        if not peer in self.peers:
            raise Exception(f"{peer} not in {self.peers}")

        epr_socket = self.context.epr_sockets[peer]

        if role == "send":
            qubit = epr_socket.create_keep()[0]
        elif role == "recv":
            qubit = epr_socket.recv_keep()[0]
        else:
            raise ValueError(f"{role} not in ['send', 'recv']")

        yield from self.connection.flush()
        return qubit

    def generate_epr_send(self, target_peer: str):
        return self.generate_epr(target_peer, role="send")

    def generate_epr_recv(self, source_peer: str):
        return self.generate_epr(source_peer, role="recv")

    def generate_noisy_epr(
            self,
            peer: str,
            role: str = "send",
            amp_x: float = np.pi/8,
            amp_y: float = np.pi/8,
            amp_z: float = np.pi/8
        ):
        epr_qubit = yield from self.generate_epr(peer, role=role)
        yield from self.add_noise(epr_qubit, amp_x=amp_x, amp_y=amp_y, amp_z=amp_z)
        return epr_qubit

    def add_noise(
            self,
            qubit: Qubit,
            amp_x: float = np.pi/8,
            amp_y: float = np.pi/8,
            amp_z: float = np.pi/8
        ):
        """ Adds noise by correlating qubit to a randomly rotated ancilla qubit """

        if not len(self.available_qubits) >= 1:
            raise Exception("An additional qubit is required for the noise model")

        noise_qubit = Qubit(self.connection)
        angle_X = abs(np.random.random()*amp_x - np.random.choice([0, 2*np.pi]))
        angle_Y = abs(np.random.random()*amp_y - np.random.choice([0, 2*np.pi]))
        angle_Z = abs(np.random.random()*amp_z - np.random.choice([0, 2*np.pi]))

        noise_qubit.rot_X(angle=angle_X)
        noise_qubit.rot_Y(angle=angle_Y)
        noise_qubit.rot_Z(angle=angle_Z)
        noise_qubit.cnot(qubit)
        noise_qubit.free()
        yield from self.connection.flush()

    def distill_epr(self, peer: str, q1: Qubit, q2: Qubit, role="send"):
        """ DEJMPS distillation (dx.doi.org/10.1103/PhysRevLett.77.2818)

        Parameters:
        - peer: str -> peer that will receive the entangled state
        - q1, q2 -> preshared EPR pairs. If q2 = None, a new pair will be generated
        """

        if role == "send":
            # Apply rotations and CNOT
            q1.rot_X(1, 1)  # Apply rot_X(pi/2) = [[1, i], [i, 1]]
            q2.rot_X(1, 1)  # Apply rot_X(pi/2) = [[1, i], [i, 1]]
            q1.cnot(q2)  # Apply CNOT gate between pair of qubits

            # Measure qubit[ind2], send value and wait for measurement from peer
            res_a = q2.measure()
            yield from self.connection.flush()
            csocket = self.csockets[peer]
            csocket.send(str(int(res_a)))
            res_b = yield from csocket.recv()
        elif role == "recv":
            # Apply rotations and CNOT
            q1.rot_X(3, 1)  # Apply rot_X(-pi/2) = [[1, -i], [-i, 1]] 
            q2.rot_X(3, 1)  # Apply rot_X(-pi/2) = [[1, -i], [-i, 1]]
            q1.cnot(q2)  # Apply CNOT gate between pair of qubits

            # Measure qubit[ind2], send value and wait for measurement from peer
            res_a = q2.measure()
            yield from self.connection.flush()
            csocket = self.csockets[peer]
            res_b = yield from csocket.recv()
            csocket.send(str(int(res_a)))
        else:
            raise ValueError(f"{role} not in ['send', 'recv']")

        # If measurements coincide, keep qubit[ind1], else erase value
        success = int(res_a) == int(res_b)
        if not success:
            q1.free()
            q1 = None
        q2 = None

        return q1

    def distill_epr_send(self, peer: str, q1: Qubit, q2: Qubit):
        return self.distill_epr(self, peer, q1, q2, role="send")

    def distill_epr_recv(self, peer: str, q1: Qubit, q2: Qubit):
        return self.distill_epr(self, peer, q1, q2, role="recv")

    def distill_epr_rounds_send(self, peer: str, rounds: int = 1, verbose=False):

        if not len(self.available_qubits) >= 2:
            raise Exception("No communication qubits available")

        ind0 = self.available_qubits[0]
        ind1 = self.available_qubits[1]

        round = 0
        attempts = 0
        while round <= rounds:
            success = yield from self.distill_epr_send(peer, ind0, ind1)
            if verbose:
                print(f"{self.name}\t round: {round}\t success: {success}")

            if not success:
                round = 0
                attempts += 1
            else:
                round += 1
        
        print(f"Completed {rounds} of distillation after {attempts} attempts")
        return ind0

    def distill_epr_rounds_recv(self, peer: str, rounds: int = 1, verbose=False):

        if not len(self.available_qubits) >= 2:
            raise Exception("No communication qubits available")

        ind0 = self.available_qubits[0]
        ind1 = self.available_qubits[1]

        round = 0
        attempts = 0
        while round <= rounds:
            success = yield from self.distill_epr_recv(peer, ind0, ind1)
            if verbose:
                print(f"{self.name}\t round: {round}\t success: {success}")

            if not success:
                round = 0
                attempts += 1
            else:
                round += 1
        return ind0

    def distributed_cnot_source(
        self,
        source_qubit: Qubit,
        target_peer: str
    ):
        """ Perform a distributed CNOT gate as source

        Parameters:
        - source_qubit: Qubit -> The qubit that is used as control source from local node
        - target_peer: str -> The connected node that will receive the control. The qubit it will
                              be applied upon will be decided by the target peer
        """

        if not target_peer in self.peers:
            raise Exception(f"{target_peer} not in {self.peers}")

        csocket = self.context.csockets[target_peer]
        epr_socket = self.context.epr_sockets[target_peer]

        # Check if there are available qubits
        if not len(self.available_qubits) > 0:
            raise Exception("No communication qubits available")

        # Send request to generate EPR pair in communication qubit
        comm_qubit = epr_socket.create_keep()[0]

        # ___________________ REMOTE CNOT PROTOCOL ___________________
        source_qubit.cnot(comm_qubit)
        local_meas = comm_qubit.measure()
        yield from self.connection.flush()

        # Send result of local measurement
        csocket.send(f"{int(local_meas)}")
        yield from self.connection.flush()

        # Receive result of remote measurement and apply Z gate if True
        rem_meas = yield from csocket.recv()
        if int(rem_meas):
            source_qubit.Z()
        yield from self.connection.flush()

        # Free up communication qubit
        comm_qubit = None
        return source_qubit

    def distributed_cnot_target(
        self,
        target_qubit: Qubit,
        source_peer: str
    ):
        """ Perform a distributed CNOT gate as target

        Parameters:
        - target_qubit: Qubit -> The qubit upon which the control will be performed from local node
        - target_peer: str -> The connected node that will be the source of control
        """

        if not source_peer in self.peers:
            raise Exception(f"{source_peer} not in {self.peers}")

        csocket = self.context.csockets[source_peer]
        epr_socket = self.context.epr_sockets[source_peer]

        # Check if there are available qubits
        if not len(self.available_qubits) > 0:
            raise Exception("No communication qubits available")

        # Accept request to generate EPR pair in communication qubit  
        comm_qubit = epr_socket.recv_keep()[0]

        # ___________________ REMOTE CNOT PROTOCOL ___________________
        comm_qubit.cnot(target_qubit)
        comm_qubit.H()
        local_meas = comm_qubit.measure()
        yield from self.connection.flush()
    
        # Receive result of remote measurement and apply X gate if True
        rem_meas = yield from csocket.recv()
        if int(rem_meas):
            target_qubit.X()
        yield from self.connection.flush()

        csocket.send(f"{int(local_meas)}")
        yield from self.connection.flush()

        # Free up communication qubit
        comm_qubit = None
        return target_qubit

    def teleport_data_send(
        self,
        source_qubit: Qubit,
        target_peer: str
    ):
        """ Perform a teledata operation (teleport the state of a qubit)

        Parameters:
        - source_qubit: Qubit -> The qubit that is used as control source from local node
        - target_peer: str -> The connected node that will receive the teleported qubit
        """

        if not target_peer in self.peers:
            raise Exception(f"{target_peer} not in {self.peers}")

        csocket = self.context.csockets[target_peer]
        epr_socket = self.context.epr_sockets[target_peer]

        comm_qubit = epr_socket.create_keep()[0]
        yield from self.connection.flush()

        source_qubit.cnot(comm_qubit)
        source_qubit.H()

        r0 = source_qubit.measure()
        r1 = comm_qubit.measure()
        yield from self.connection.flush()

        msg = f"{int(r0)},{int(r1)}"
        csocket.send(msg)

        source_qubit = None
        comm_qubit = None
        return source_qubit

    def teleport_data_recv(
        self,
        source_peer: str
    ):
        """ Perform a teledata operation (teleport the state of a qubit)

        Parameters:
        - source_peer: str -> The connected node that will send the teleported qubit

        Returns:
        - qubit: Qubit
        """

        if not source_peer in self.peers:
            raise Exception(f"{source_peer} not in {self.peers}")

        csocket = self.context.csockets[source_peer]
        epr_socket = self.context.epr_sockets[source_peer]

        comm_qubit = epr_socket.recv_keep()[0]
        yield from self.connection.flush()

        msg = yield from csocket.recv()
        r0, r1 = msg.split(",")

        if int(r1):
            comm_qubit.X()
        if int(r0):
            comm_qubit.Z()

        yield from self.connection.flush()

        return comm_qubit

    def epr_to_ghz_source(
        self,
        epr_pairs: List[Qubit],
        peers: List[str],
        keep_entangled: bool = True
    ):
        """ Generates GHZ state from preformed entangled Bell pairs 

        Parameters:
        - epr_pairs: list of preformed pairs
        - peers: list of peers
        """

        for peer, epr in zip(peers[1:], epr_pairs[1:]):
            if not peer in self.peers:
                raise Exception(f"{peer} not in {self.peers}")

            csocket = self.context.csockets[peer]

            yield from self.connection.flush()
            epr_pairs[0].cnot(epr)
            res = epr.measure()
            yield from self.connection.flush()

            msg = f"{int(res)}"
            csocket.send(msg)

            # Free up measured qubit
            epr = None

        csocket = self.context.csockets[peers[0]]

        if not keep_entangled:
             # Disentangle router qubit from the rest by measuring in X basis
            epr_pairs[0].H()
            res = epr_pairs[0].measure()
            yield from self.connection.flush()

            # Send result of measurement in X basis to first GHZ node
            if not int(res):
                csocket.send('+')
            else:
                csocket.send('-')

            # Free up measured qubit
            epr_pairs = None
        else:

            # Send '0' to do nothing and clear socket
            csocket.send('0')

    def epr_to_ghz_target(
        self,
        epr_pair: Qubit,
        source_peer: str,
    ):
        """ Receive instruction to apply X gate based on measurement to generate Bell pair 

        Parameters:
        - epr_pair: Preformed entangled pair
        - source_peer: The peer that will perform the measurements of the preformed pairs
        """

        csocket = self.context.csockets[source_peer]

        # Receive result of remote measurement
        msg = yield from csocket.recv()

        # If a X basis measurement was performed and result was -, apply Z
        if msg == '-':
            epr_pair.Z()

        # If a Z basis measurement was performed and result was 1, apply X
        if msg == '1':
            epr_pair.X()
        yield from self.connection.flush()     

    def cat_entangle_source(self, qubit: Qubit, peer: str):
        epr_socket = self.context.epr_sockets[peer]
        csocket = self.context.csockets[peer]
        epr = epr_socket.create_keep()[0]

        qubit.cnot(epr)
        r = epr.measure()

        yield from self.connection.flush()

        msg = f"{int(r)}"
        csocket.send(msg)

        yield from self.connection.flush()

    def cat_entangle_target(self, peer: str):
        
        epr_socket = self.context.epr_sockets[peer]
        
        csocket = self.context.csockets[peer]

        epr = epr_socket.recv_keep()[0]
        
        yield from self.connection.flush()   ############################################################
        
        m = yield from csocket.recv()
        
        if int(m):
            epr.X()
            
        return epr

    def cat_disentangle_source(self, qubit: Qubit, peer: str):
        csocket = self.context.csockets[peer]

        yield from self.connection.flush()

        m = yield from csocket.recv()

        if int(m):
            qubit.Z()

    def cat_disentangle_target(self, epr: Qubit, peer: str):
        csocket = self.context.csockets[peer]

        epr.H()
        r = epr.measure()

        yield from self.connection.flush()

        msg = f"{int(r)}"
        csocket.send(msg)

        yield from self.connection.flush()

    def QFT(
        self,
        qubits: List[Qubit],
        reorder: bool = True,
        verbose: bool = False
    ):
        """ Perform quantum fourier transform (locally)

        Parameters:
        - qubits: list of local qubits
        - reorder: if True, performs normal QFT, else skips the reordering of the qubits 
        """

        num_qubits = len(qubits)

        if reorder:
            for i in range(len(qubits)//2):
                if verbose:
                    print(f"{self.name}: SWAP({i}, {num_qubits-i-1})")
                SWAP(qubits[i], qubits[num_qubits-i-1])

        # First H
        if verbose:
            print(f"{self.name}: H{0}")
        qubits[0].H()

        for i in range(1, num_qubits):

            for j in range(0, i):
                if verbose:
                    print(f"{self.name}: CP({i}, {j}, θ = π/{2**(i-j)})")
                CP(qubits[i], qubits[j], angle= np.pi/(2**(i-j)))

            # Apply H per layer
            if verbose:
                print(f"{self.name}: H{i}")
            qubits[i].H()


    def inverse_QFT(
        self,
        qubits: List[Qubit],
        reorder: bool = True,
        verbose: bool = False
    ):
        """ Perform inverse quantum fourier transform (locally)

        Parameters:
        - qubits: list of local qubits
        - reorder: if True, performs normal inverse QFT, else skips the reordering of the qubits 
        """

        num_qubits = len(qubits)

        for i in range(num_qubits-1, 0, -1):
            # Apply H per layer
            if verbose:
                print(f"{self.name}: H{i}")
            qubits[i].H()

            for j in range(i-1, -1, -1):
                if verbose:
                    print(f"{self.name}: CP({i}, {j}, θ = -π/{(2**((i-j)))})")
                CP(qubits[i], qubits[j], angle= -np.pi/(2**((i-j))))

        # Final H
        if verbose:
            print(f"{self.name}: H{0}")
        qubits[0].H()

        if reorder:
            for i in range(len(qubits)//2):
                if verbose:
                    print(f"{self.name}: SWAP({i}, {num_qubits-i-1})")
                SWAP(qubits[i], qubits[num_qubits-i-1])

    def inverse_QFT_dist_source(self, qubits: List[Qubit], peer: str, verbose: bool = False):
        self.inverse_QFT(qubits, reorder=False,verbose=verbose)
        yield from self._inverse_QFT_dist_intermediate_layer_source(qubits, peer, verbose=verbose)

    def inverse_QFT_dist_target(self, qubits: List[Qubit], peer: str, verbose: bool = False):
        yield from self._inverse_QFT_dist_intermediate_layer_target(qubits, peer, verbose=verbose)
        self.inverse_QFT(qubits, reorder=False,verbose=verbose)

    def _inverse_QFT_dist_intermediate_layer_source(
            self,
            qubits: List[Qubit],
            peer: str,
            verbose: bool = False
        ):
        
        csocket = self.context.csockets[peer]
        msg = f"{int(len(qubits))}"
        csocket.send(msg)

        yield from self.connection.flush()

        for q in qubits:
            yield from self.cat_entangle_source(q, peer)
            yield from self.cat_disentangle_source(q, peer)

    def _inverse_QFT_dist_intermediate_layer_target(
            self,
            qubits: List[Qubit],
            peer: str,
            verbose: bool = False
        ):

        csocket = self.context.csockets[peer]
        msg = yield from csocket.recv()

        num_peer_qubits = int(msg)
        num_own_qubits = len(qubits)
        
        for i in range(num_peer_qubits):

            epr = yield from self.cat_entangle_target(peer) ############################################################

            for j, qubit in enumerate(qubits):
                if verbose:
                    print(f"{self.name}: CP({i+num_own_qubits}, {j}, θ = -π/{2**(i+num_own_qubits-j)})")
                CP(epr, qubit, angle=-np.pi/2**(i+num_own_qubits-j))

            yield from self.cat_disentangle_target(epr, peer)
