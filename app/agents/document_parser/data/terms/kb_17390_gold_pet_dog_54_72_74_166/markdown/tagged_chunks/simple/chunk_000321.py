from langchain_core.documents import Document

chunk = Document(
    page_content=('약의 소멸) 및 제36조(중도인출)는 제외합니다.82 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)# 11. '
 '골절수술비제1조(보험금의 지급사유)\n'
 '회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 직접결과로써 【별표6】(골\n'
 '절분류표)에서 정한 골절로 진단확정 후 치료를 직접적인 목적으로 수술을 받은 경\n'
 '우 이 특별약관의 보험가입금액을 골절수술비로 보험수익자에게 매 사고시마다 지급# 합니다.- 제 2조(보험금 지급에 관한 세부규정)\n'
 '- \uf000 제1조(보험금의 지급사유)의 골절수술비는 같은 상해로 두 종류 이상의 골절수술'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000321',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
