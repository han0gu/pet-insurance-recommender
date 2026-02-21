from langchain_core.documents import Document

chunk = Document(
    page_content=('습니다. 제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는\n'
 '의료비용은 회사가 전액 부담합니다.제3조(보험금을 지급하지 않는 사유)회사는 아래의 사유로 인한손해는 보상하지 않습니다.1. 계약자, '
 '피보험자, 이들의 가족 또는 사용인의고의 또는 중대한 과실- 2. 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 '
 '이들과 유사한\n'
 '- 사태\n'
 '- 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변\n'
 '- 4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성 또는 그'),
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
 'indexing': {'chunk_id': 'chunk_000630',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
