use std::collections::HashMap;
use std::convert::TryInto;

use anyhow::Result;
use num_traits::One;
use num_traits::Zero;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
use twenty_first::shared_math::other::log_2_floor;
use twenty_first::shared_math::tip5;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::shared_math::tip5::Tip5State;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::Domain;

use triton_opcodes::instruction::AnInstruction::*;
use triton_opcodes::instruction::Instruction;
use triton_opcodes::ord_n::Ord16::*;
use triton_opcodes::program::Program;

use crate::error::vm_err;
use crate::error::vm_fail;
use crate::error::InstructionError::InstructionPointerOverflow;
use crate::error::InstructionError::*;
use crate::op_stack::OpStack;
use crate::vm::VMOutput::*;

#[derive(Debug, Default, Clone)]
pub struct VMState<'pgm> {
    // Memory
    /// The **program memory** stores the instructions (and their arguments) of the program
    /// currently being executed by Triton VM. It is read-only.
    pub program: &'pgm [Instruction],

    /// The read-write **random-access memory** allows Triton VM to store arbitrary data.
    pub ram: HashMap<BFieldElement, BFieldElement>,

    /// The **Op-stack memory** stores Triton VM's entire operational stack.
    pub op_stack: OpStack,

    /// The **Jump-stack memory** stores the entire jump stack.
    pub jump_stack: Vec<(BFieldElement, BFieldElement)>,

    // Registers
    /// Number of cycles the program has been running for
    pub cycle_count: u32,

    /// Current instruction's address in program memory
    pub instruction_pointer: usize,

    /// The instruction that was executed last
    pub previous_instruction: BFieldElement,

    /// RAM pointer
    pub ramp: u64,

    /// The current state of the one, global Sponge that can be manipulated using instructions
    /// `AbsorbInit`, `Absorb`, and `Squeeze`. Instruction `AbsorbInit` resets the state prior to
    /// absorbing.
    /// Note that this is the _full_ state, including capacity. The capacity should never be
    /// exposed outside of the VM.
    pub sponge_state: [BFieldElement; tip5::STATE_SIZE],

    // Bookkeeping
    /// Indicates whether the terminating instruction `halt` has been executed.
    pub halting: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum VMOutput {
    /// Trace output from `write_io`
    WriteOutputSymbol(BFieldElement),

    /// Trace of the state registers for hash coprocessor table when executing instruction `hash`
    /// or any of the Sponge instructions `absorb_init`, `absorb`, `squeeze`.
    /// One row per round in the Tip5 permutation.
    Tip5Trace(
        Instruction,
        Box<[[BFieldElement; tip5::STATE_SIZE]; 1 + tip5::NUM_ROUNDS]>,
    ),

    /// Executed u32 instruction as well as its left-hand side and right-hand side
    U32TableEntries(Vec<(Instruction, BFieldElement, BFieldElement)>),
}

impl<'pgm> VMState<'pgm> {
    /// Create initial `VMState` for a given `program`
    ///
    /// Since `program` is read-only across individual states, and multiple
    /// inner helper functions refer to it, a read-only reference is kept in
    /// the struct.
    pub fn new(program: &'pgm Program) -> Self {
        let program = &program.instructions;
        Self {
            program,
            ..VMState::default()
        }
    }

    /// Given a state, compute `(next_state, vm_output)`.
    pub fn step(
        &self,
        stdin: &mut Vec<BFieldElement>,
        secret_in: &mut Vec<BFieldElement>,
    ) -> Result<(VMState<'pgm>, Option<VMOutput>)> {
        let mut next_state = self.clone();
        next_state
            .step_mut(stdin, secret_in)
            .map(|vm_output| (next_state, vm_output))
    }

    /// Perform the state transition as a mutable operation on `self`.
    pub fn step_mut(
        &mut self,
        stdin: &mut Vec<BFieldElement>,
        secret_in: &mut Vec<BFieldElement>,
    ) -> Result<Option<VMOutput>> {
        // All instructions increase the cycle count
        self.cycle_count += 1;
        let mut vm_output = None;
        self.previous_instruction = match self.current_instruction() {
            Ok(instruction) => instruction.opcode_b(),
            // trying to read past the end of the program doesn't change the previous instruction
            Err(_) => self.previous_instruction,
        };

        match self.current_instruction()? {
            Pop => {
                self.op_stack.pop()?;
                self.instruction_pointer += 1;
            }

            Push(arg) => {
                self.op_stack.push(arg);
                self.instruction_pointer += 2;
            }

            Divine(_) => {
                let elem = secret_in.remove(0);
                self.op_stack.push(elem);
                self.instruction_pointer += 1;
            }

            Dup(arg) => {
                let elem = self.op_stack.safe_peek(arg);
                self.op_stack.push(elem);
                self.instruction_pointer += 2;
            }

            Swap(arg) => {
                // st[0] ... st[n] -> st[n] ... st[0]
                self.op_stack.safe_swap(arg);
                self.instruction_pointer += 2;
            }

            Nop => {
                self.instruction_pointer += 1;
            }

            Skiz => {
                let elem = self.op_stack.pop()?;
                self.instruction_pointer += if elem.is_zero() {
                    let next_instruction = self.next_instruction()?;
                    1 + next_instruction.size()
                } else {
                    1
                };
            }

            Call(addr) => {
                let o_plus_2 = self.instruction_pointer as u32 + 2;
                let pair = (BFieldElement::new(o_plus_2 as u64), addr);
                self.jump_stack.push(pair);
                self.instruction_pointer = addr.value() as usize;
            }

            Return => {
                let (orig_addr, _dest_addr) = self.jump_stack_pop()?;
                self.instruction_pointer = orig_addr.value() as usize;
            }

            Recurse => {
                let (_orig_addr, dest_addr) = self.jump_stack_peek()?;
                self.instruction_pointer = dest_addr.value() as usize;
            }

            Assert => {
                let elem = self.op_stack.pop()?;
                if !elem.is_one() {
                    return vm_err(AssertionFailed(
                        self.instruction_pointer,
                        self.cycle_count,
                        elem,
                    ));
                }
                self.instruction_pointer += 1;
            }

            Halt => {
                self.halting = true;
                self.instruction_pointer += 1;
            }

            ReadMem => {
                let ramp = self.op_stack.safe_peek(ST0);
                let ramv = self.memory_get(&ramp);
                self.op_stack.push(ramv);
                self.ramp = ramp.value();
                self.instruction_pointer += 1;
            }

            WriteMem => {
                let ramp = self.op_stack.safe_peek(ST1);
                let ramv = self.op_stack.pop()?;
                self.ramp = ramp.value();
                self.ram.insert(ramp, ramv);
                self.instruction_pointer += 1;
            }

            Hash => {
                let to_hash = self.op_stack.pop_n::<{ tip5::RATE }>()?;
                let mut hash_input = Tip5State::new(Domain::FixedLength);
                hash_input.state[..tip5::RATE].copy_from_slice(&to_hash);
                let tip5_trace = Tip5::trace(&mut hash_input);
                let hash_output = &tip5_trace[tip5_trace.len() - 1][0..DIGEST_LENGTH];

                for i in (0..DIGEST_LENGTH).rev() {
                    self.op_stack.push(hash_output[i]);
                }
                for _ in 0..DIGEST_LENGTH {
                    self.op_stack.push(BFieldElement::zero());
                }

                vm_output = Some(Tip5Trace(Hash, Box::new(tip5_trace)));
                self.instruction_pointer += 1;
            }

            AbsorbInit | Absorb => {
                // fetch top elements but don't alter the stack
                let to_absorb = self.op_stack.pop_n::<{ tip5::RATE }>()?;
                for i in (0..tip5::RATE).rev() {
                    self.op_stack.push(to_absorb[i]);
                }

                if self.current_instruction()? == AbsorbInit {
                    self.sponge_state = Tip5State::new(Domain::VariableLength).state;
                }
                self.sponge_state[..tip5::RATE].copy_from_slice(&to_absorb);
                let tip5_trace = Tip5::trace(&mut Tip5State {
                    state: self.sponge_state,
                });
                self.sponge_state = tip5_trace.last().unwrap().to_owned();

                vm_output = Some(Tip5Trace(self.current_instruction()?, Box::new(tip5_trace)));
                self.instruction_pointer += 1;
            }

            Squeeze => {
                let _ = self.op_stack.pop_n::<{ tip5::RATE }>()?;
                for i in (0..tip5::RATE).rev() {
                    self.op_stack.push(self.sponge_state[i]);
                }
                let tip5_trace = Tip5::trace(&mut Tip5State {
                    state: self.sponge_state,
                });
                self.sponge_state = tip5_trace.last().unwrap().to_owned();

                vm_output = Some(Tip5Trace(Squeeze, Box::new(tip5_trace)));
                self.instruction_pointer += 1;
            }

            DivineSibling => {
                self.divine_sibling(secret_in)?;
                self.instruction_pointer += 1;
            }

            AssertVector => {
                if !self.assert_vector() {
                    return vm_err(AssertionFailed(
                        self.instruction_pointer,
                        self.cycle_count,
                        self.op_stack
                            .peek(0)
                            .expect("Could not unwrap top of stack."),
                    ));
                }
                self.instruction_pointer += 1;
            }

            Add => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(lhs + rhs);
                self.instruction_pointer += 1;
            }

            Mul => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(lhs * rhs);
                self.instruction_pointer += 1;
            }

            Invert => {
                let elem = self.op_stack.pop()?;
                if elem.is_zero() {
                    return vm_err(InverseOfZero);
                }
                self.op_stack.push(elem.inverse());
                self.instruction_pointer += 1;
            }

            Eq => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop()?;
                self.op_stack.push(Self::eq(lhs, rhs));
                self.instruction_pointer += 1;
            }

            Split => {
                let elem = self.op_stack.pop()?;
                let lo = BFieldElement::new(elem.value() & 0xffff_ffff);
                let hi = BFieldElement::new(elem.value() >> 32);
                self.op_stack.push(hi);
                self.op_stack.push(lo);
                self.instruction_pointer += 1;
                let u32_table_entry = (Split, lo, hi);
                vm_output = Some(U32TableEntries(vec![u32_table_entry]));
            }

            Lt => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let lt = BFieldElement::new((lhs < rhs) as u64);
                self.op_stack.push(lt);
                self.instruction_pointer += 1;
                let u32_table_entry = (Lt, (lhs as u64).into(), (rhs as u64).into());
                vm_output = Some(U32TableEntries(vec![u32_table_entry]));
            }

            And => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let and = BFieldElement::new((lhs & rhs) as u64);
                self.op_stack.push(and);
                self.instruction_pointer += 1;
                let u32_table_entry = (And, (lhs as u64).into(), (rhs as u64).into());
                vm_output = Some(U32TableEntries(vec![u32_table_entry]));
            }

            Xor => {
                let lhs = self.op_stack.pop_u32()?;
                let rhs = self.op_stack.pop_u32()?;
                let xor = BFieldElement::new((lhs ^ rhs) as u64);
                self.op_stack.push(xor);
                self.instruction_pointer += 1;
                // Triton VM uses the following equality to compute the results of both the `and`
                // and `xor` instruction using the u32 coprocessor's `and` capability:
                // a ^ b = a + b - 2 · (a & b)
                let u32_table_entry = (And, (lhs as u64).into(), (rhs as u64).into());
                vm_output = Some(U32TableEntries(vec![u32_table_entry]));
            }

            Log2Floor => {
                let lhs = self.op_stack.pop_u32()?;
                if lhs.is_zero() {
                    return vm_err(LogarithmOfZero);
                }
                let l2f = BFieldElement::new(log_2_floor(lhs as u128));
                self.op_stack.push(l2f);
                self.instruction_pointer += 1;
                let u32_table_entry = (Log2Floor, (lhs as u64).into(), BFIELD_ZERO);
                vm_output = Some(U32TableEntries(vec![u32_table_entry]));
            }

            Pow => {
                let lhs = self.op_stack.pop()?;
                let rhs = self.op_stack.pop_u32()?;
                let pow = lhs.mod_pow(rhs as u64);
                self.op_stack.push(pow);
                self.instruction_pointer += 1;
                let u32_table_entry = (Pow, lhs, (rhs as u64).into());
                vm_output = Some(U32TableEntries(vec![u32_table_entry]));
            }

            Div => {
                let numer = self.op_stack.pop_u32()?;
                let denom = self.op_stack.pop_u32()?;
                if denom.is_zero() {
                    return vm_err(DivisionByZero);
                }
                let quot = BFieldElement::new((numer / denom) as u64);
                let rem = BFieldElement::new((numer % denom) as u64);
                self.op_stack.push(quot);
                self.op_stack.push(rem);
                self.instruction_pointer += 1;
                let u32_table_entry_0 = (Lt, rem, (denom as u64).into());
                let u32_table_entry_1 = (Split, (numer as u64).into(), quot);
                vm_output = Some(U32TableEntries(vec![u32_table_entry_0, u32_table_entry_1]));
            }

            PopCount => {
                let lhs = self.op_stack.pop_u32()?;
                let pop_count = BFieldElement::new(lhs.count_ones() as u64);
                self.op_stack.push(pop_count);
                self.instruction_pointer += 1;
                let u32_table_entry = (PopCount, (lhs as u64).into(), BFIELD_ZERO);
                vm_output = Some(U32TableEntries(vec![u32_table_entry]));
            }

            XxAdd => {
                let lhs: XFieldElement = self.op_stack.pop_x()?;
                let rhs: XFieldElement = self.op_stack.safe_peek_x();
                self.op_stack.push_x(lhs + rhs);
                self.instruction_pointer += 1;
            }

            XxMul => {
                let lhs: XFieldElement = self.op_stack.pop_x()?;
                let rhs: XFieldElement = self.op_stack.safe_peek_x();
                self.op_stack.push_x(lhs * rhs);
                self.instruction_pointer += 1;
            }

            XInvert => {
                let elem: XFieldElement = self.op_stack.pop_x()?;
                if elem.is_zero() {
                    return vm_err(InverseOfZero);
                }
                self.op_stack.push_x(elem.inverse());
                self.instruction_pointer += 1;
            }

            XbMul => {
                let lhs: BFieldElement = self.op_stack.pop()?;
                let rhs: XFieldElement = self.op_stack.pop_x()?;
                self.op_stack.push_x(lhs.lift() * rhs);
                self.instruction_pointer += 1;
            }

            WriteIo => {
                vm_output = Some(WriteOutputSymbol(self.op_stack.pop()?));
                self.instruction_pointer += 1;
            }

            ReadIo => {
                let in_elem = stdin.remove(0);
                self.op_stack.push(in_elem);
                self.instruction_pointer += 1;
            }
        }

        // Check that no instruction left the OpStack with too few elements
        if self.op_stack.is_too_shallow() {
            return vm_err(OpStackTooShallow);
        }

        Ok(vm_output)
    }

    fn eq(lhs: BFieldElement, rhs: BFieldElement) -> BFieldElement {
        if lhs == rhs {
            BFieldElement::one()
        } else {
            BFieldElement::zero()
        }
    }

    fn current_instruction(&self) -> Result<Instruction> {
        self.program
            .get(self.instruction_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(self.instruction_pointer)))
            .copied()
    }

    // Return the next instruction on the tape, skipping arguments
    //
    // Note that this is not necessarily the next instruction to execute,
    // since the current instruction could be a jump, but it is either
    // program[ip + 1] or program[ip + 2] depending on whether the current
    // instruction takes an argument or not.
    fn next_instruction(&self) -> Result<Instruction> {
        let ci = self.current_instruction()?;
        let ci_size = ci.size();
        let ni_pointer = self.instruction_pointer + ci_size;
        self.program
            .get(ni_pointer)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(ni_pointer)))
            .copied()
    }

    fn _next_next_instruction(&self) -> Result<Instruction> {
        let cur_size = self.current_instruction()?.size();
        let next_size = self.next_instruction()?.size();
        self.program
            .get(self.instruction_pointer + cur_size + next_size)
            .ok_or_else(|| vm_fail(InstructionPointerOverflow(self.instruction_pointer)))
            .copied()
    }

    fn jump_stack_pop(&mut self) -> Result<(BFieldElement, BFieldElement)> {
        self.jump_stack
            .pop()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn jump_stack_peek(&mut self) -> Result<(BFieldElement, BFieldElement)> {
        self.jump_stack
            .last()
            .copied()
            .ok_or_else(|| vm_fail(JumpStackTooShallow))
    }

    fn memory_get(&self, mem_addr: &BFieldElement) -> BFieldElement {
        self.ram
            .get(mem_addr)
            .copied()
            .unwrap_or_else(BFieldElement::zero)
    }

    fn assert_vector(&self) -> bool {
        for i in 0..DIGEST_LENGTH {
            // Safe as long as 2 * DIGEST_LEN <= OP_STACK_REG_COUNT
            let lhs = i.try_into().expect("Digest element position (lhs)");
            let rhs = (i + DIGEST_LENGTH)
                .try_into()
                .expect("Digest element position (rhs)");

            if self.op_stack.safe_peek(lhs) != self.op_stack.safe_peek(rhs) {
                return false;
            }
        }
        true
    }

    pub fn read_word(&self) -> Result<Option<BFieldElement>> {
        let current_instruction = self.current_instruction()?;
        if matches!(current_instruction, ReadIo) {
            Ok(Some(self.op_stack.safe_peek(ST0)))
        } else {
            Ok(None)
        }
    }

    fn divine_sibling(&mut self, secret_in: &mut Vec<BFieldElement>) -> Result<()> {
        // st0-st4
        let _ = self.op_stack.pop_n::<{ DIGEST_LENGTH }>()?;

        // st5-st9
        let known_digest = self.op_stack.pop_n::<{ DIGEST_LENGTH }>()?;

        // st10
        let node_index_elem: BFieldElement = self.op_stack.pop()?;
        let node_index: u32 = node_index_elem
            .try_into()
            .unwrap_or_else(|_| panic!("{node_index_elem:?} is not a u32"));

        // nondeterministic guess, flipped
        let sibling_digest: [BFieldElement; DIGEST_LENGTH] = {
            let mut tmp = [
                secret_in.remove(0),
                secret_in.remove(0),
                secret_in.remove(0),
                secret_in.remove(0),
                secret_in.remove(0),
            ];
            tmp.reverse();
            tmp
        };

        // least significant bit
        let hv0 = node_index % 2;

        // push new node index
        // st10
        self.op_stack
            .push(BFieldElement::new(node_index as u64 >> 1));

        // push 2 digests, in correct order
        // Correct order means the following:
        //
        // | sponge | stack | digest element | hv0 == 0 | hv0 == 1 |
        // |--------|-------|----------------|----------|----------|
        // | r0     | st0   | left0          | known0   | sibling0 |
        // | r1     | st1   | left1          | known1   | sibling1 |
        // | r2     | st2   | left2          | known2   | sibling2 |
        // | r3     | st3   | left3          | known3   | sibling3 |
        // | r4     | st4   | left4          | known4   | sibling4 |
        // | r5     | st5   | right0         | sibling0 | known0   |
        // | r6     | st6   | right1         | sibling1 | known1   |
        // | r7     | st7   | right2         | sibling2 | known2   |
        // | r8     | st8   | right3         | sibling3 | known3   |
        // | r9     | st9   | right4         | sibling4 | known4   |

        let (top_digest, runner_up) = if hv0 == 0 {
            (known_digest, sibling_digest)
        } else {
            (sibling_digest, known_digest)
        };

        for digest_element in runner_up.iter().rev() {
            self.op_stack.push(*digest_element);
        }

        for digest_element in top_digest.iter().rev() {
            self.op_stack.push(*digest_element);
        }

        Ok(())
    }
}

/// Similar to [`run`], but also returns a [`Vec`] of [`VMState`]s, one for each step of the VM.
/// On premature termination of the VM, returns all [`VMState`]s and output for the execution up
/// to the point of failure.
///
/// The VM's initial state is either the provided `initial_state`, or a new [`VMState`] if
/// `initial_state` is `None`. The initial state is not included in the returned [`Vec`] of
/// [`VMState`]s. The initial state is the state of the VM before the first instruction is
/// executed. The initial state must contain the same program as provided by parameter `program`,
/// else the method will panic.
///
/// If `num_cycles_to_execute` is `Some(number_of_cycles)`, the VM will execute at most
/// `number_of_cycles` cycles. If `num_cycles_to_execute` is `None`, the VM will execute until
/// it halts.
///
/// See also [`simulate`].
pub fn debug<'pgm>(
    program: &'pgm Program,
    mut stdin: Vec<BFieldElement>,
    mut secret_in: Vec<BFieldElement>,
    initial_state: Option<VMState<'pgm>>,
    num_cycles_to_execute: Option<u32>,
) -> (
    Vec<VMState<'pgm>>,
    Vec<BFieldElement>,
    Option<anyhow::Error>,
) {
    let mut states = vec![];
    let mut stdout = vec![];
    let mut current_state = initial_state.unwrap_or(VMState::new(program));
    let max_cycles = match num_cycles_to_execute {
        Some(number_of_cycles) => current_state.cycle_count + number_of_cycles,
        None => u32::MAX,
    };

    assert_eq!(
        current_state.program, program.instructions,
        "The (optional) initial state must be for the given program."
    );

    while !current_state.halting && current_state.cycle_count < max_cycles {
        states.push(current_state.clone());
        let step = current_state.step(&mut stdin, &mut secret_in);
        let (next_state, vm_output) = match step {
            Err(err) => return (states, stdout, Some(err)),
            Ok((next_state, vm_output)) => (next_state, vm_output),
        };

        if let Some(WriteOutputSymbol(written_word)) = vm_output {
            stdout.push(written_word);
        }
        current_state = next_state;
    }

    (states, stdout, None)
}

/// Run Triton VM on the given [`Program`] with the given public and secret input.
///
/// See also [`debug`].
pub fn run(
    program: &Program,
    mut stdin: Vec<BFieldElement>,
    mut secret_in: Vec<BFieldElement>,
) -> Result<Vec<BFieldElement>, anyhow::Error> {
    let mut state = VMState::new(program);
    let mut stdout = vec![];

    while !state.halting {
        let vm_output = state.step_mut(&mut stdin, &mut secret_in)?;
        if let Some(WriteOutputSymbol(written_word)) = vm_output {
            stdout.push(written_word);
        }
    }

    Ok(stdout)
}
