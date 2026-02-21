from langchain_core.documents import Document

chunk = Document(
    page_content=('지급합니다. 다만, 증가된 위험과 관계없이 발생한 보험금\n'
 '지급사유에 관해서는 원래대로 지급합니다.# 【비례보상 예시】보험기간 중 직업의 변경으로 위험이 증가(상해급수 1급\n'
 '→ 2급)되었으나, 이를 회사에 알리지 않고 변경전 보험\n'
 '료를 계속 납입하던 중 상해사망 사고가 발생한 경우- ∙ 상해사망 가입금액 : 1억원\n'
 '- ∙ 상해사망 보험요율 : 1급 0.3, 2급 0.5\n'
 '- ⇒ 고객이 수령하는 상해사망 보험금 = 1억원 × (0.3\n'
 '- ÷ 0.5) = 6천만원\n'
 '\uf000 계약자 또는 피보험자가 고의 또는 중대한 과실로 제1항'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000052',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
