from langchain_core.documents import Document

chunk = Document(
    page_content=('【비례보상 예시】\n'
 '보험기간 중 직업의 변경으로 위험이 증가(상해급수 1급 → 2급)되었으나, 이를 회사에 알리지 않고 변경전 보험 료를 계속 납입하던 중 '
 '상해사망 사고가 발생한 경우\n'
 '∙ 상해사망 가입금액 : 1억원 ∙ 상해사망 보험요율 : 1급 0.3, 2급 0.5 ⇒ 고객이 수령하는 상해사망 보험금 = 1억원 × '
 '(0.3 ÷ 0.5) = 6천만원'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 60},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000062',
              'chunk_char_len': 195,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
