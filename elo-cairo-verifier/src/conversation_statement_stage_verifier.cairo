use starknet::ContractAddress;

#[starknet::interface]
pub trait IConversationStatementStageVerifier<TContractState> {
    fn verify_and_attest(
        ref self: TContractState,
        session: ContractAddress,
        session_id: u64,
        stage_index: u32,
        statement_felts: Array<felt252>,
    ) -> felt252;
    fn compute_statement_hash(self: @TContractState, statement_felts: Array<felt252>) -> felt252;
}

#[starknet::contract]
pub mod ConversationStatementStageVerifierContract {
    use core::poseidon::poseidon_hash_span;
    use starknet::ContractAddress;
    use crate::statement_verification_session::{
        IStatementVerificationSessionDispatcher, IStatementVerificationSessionDispatcherTrait,
        StatementSessionInfo,
    };
    use super::IConversationStatementStageVerifier;

    const MIN_SECURITY_BITS: u32 = 160;
    const DOMAIN_BATCH: felt252 = 0x43424154; // "CBAT"
    const CONVERSATION_STATEMENT_VERSION: felt252 = 1;
    const CONVERSATION_STATEMENT_FELTS: u32 = 19;

    #[storage]
    struct Storage {}

    #[abi(embed_v0)]
    impl ConversationStatementStageVerifierImpl of IConversationStatementStageVerifier<
        ContractState,
    > {
        fn verify_and_attest(
            ref self: ContractState,
            session: ContractAddress,
            session_id: u64,
            stage_index: u32,
            statement_felts: Array<felt252>,
        ) -> felt252 {
            let session_dispatcher = IStatementVerificationSessionDispatcher {
                contract_address: session,
            };
            let info = session_dispatcher.get_session(session_id);
            let statement_hash = assert_statement_matches_session(statement_felts.span(), info);
            session_dispatcher.attest_stage(session_id, stage_index, statement_hash);
            statement_hash
        }

        fn compute_statement_hash(
            self: @ContractState, statement_felts: Array<felt252>,
        ) -> felt252 {
            assert_canonical_statement(statement_felts.span())
        }
    }

    fn assert_statement_matches_session(
        statement_felts: Span<felt252>, info: StatementSessionInfo,
    ) -> felt252 {
        let statement_hash = assert_canonical_statement(statement_felts);
        assert!(*statement_felts.at(2) == info.model_id, "statement model mismatch");
        assert!(*statement_felts.at(3) == info.program_hash, "statement program mismatch");
        let statement_security: u32 = (*statement_felts.at(18)).try_into().unwrap();
        assert!(statement_security == info.security_bits, "statement security mismatch");
        assert!(statement_hash == info.statement_hash, "statement hash mismatch");
        statement_hash
    }

    fn assert_canonical_statement(statement_felts: Span<felt252>) -> felt252 {
        assert!(statement_felts.len() == CONVERSATION_STATEMENT_FELTS, "statement felts len");
        assert!(*statement_felts.at(0) == DOMAIN_BATCH, "statement domain mismatch");
        assert!(
            *statement_felts.at(1) == CONVERSATION_STATEMENT_VERSION, "statement version mismatch",
        );
        assert!(*statement_felts.at(2) != 0, "statement model missing");
        assert!(*statement_felts.at(3) != 0, "statement program missing");
        assert!(*statement_felts.at(4) != 0, "statement circuit missing");
        assert!(*statement_felts.at(5) != 0, "statement weight missing");
        assert!(*statement_felts.at(6) != 0, "statement policy missing");
        assert!(*statement_felts.at(14) != 0, "statement conversations missing");
        assert!(*statement_felts.at(15) != 0, "statement steps missing");

        let statement_security: u32 = (*statement_felts.at(18)).try_into().unwrap();
        assert!(statement_security >= MIN_SECURITY_BITS, "statement security below 160");

        poseidon_hash_span(statement_felts)
    }
}
