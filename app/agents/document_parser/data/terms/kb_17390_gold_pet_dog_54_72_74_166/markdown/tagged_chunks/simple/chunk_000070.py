from langchain_core.documents import Document

chunk = Document(
    page_content=('∙상해사망 보험요율 : 1급 0.3, 2급 0.5 → 고객이 수령하는 상해사망 보험금 = 1억원 × (0.3 ÷ 0.5) = 6천만원 '
 '\uf000 계약자 또는 피보험자가 고의 또는 중대한 과실로 제1항 각 호의 변경사실을 회사 | 회사에 알리지 않고 변경전 보험료를 계속 '
 '납입하던 중 상해사망 사고가 발생한 경우 ∙상해사망 가입금액 : 1억원 ∙상해사망 보험요율 : 1급 0.3, 2급 0.5 → 고객이 '
 '수령하는 상해사망 보험금 = 1억원 × (0.3 ÷ 0.5) = 6천만원 \uf000 계약자 또는 피보험자가 고의 또는 중대한 과실로 '
 '제1항 각 호의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000070',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
