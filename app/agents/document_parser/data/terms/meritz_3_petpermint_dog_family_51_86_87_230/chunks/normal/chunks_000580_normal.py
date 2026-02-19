from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 알릴의무 관련 용어\n'
 '용어 | 정의\n'
 '중요한 사항 | 계약 전 알릴 의무와 관련하여 회사가 그 사실 을 알았더라면 계약의 청약을 거절하거나 보험 가입금액 한도 제한, 일부 '
 '보장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합니다.\n'
 '\uf000 보상 관련 용어\n'
 '용어 | 정의\n'
 '사고 | 사고라 함은 급격하게 발생하는 것을 포함하여 위험이 서서히, 계속적, 반복적 또는 누적적으 로 노출되어 그 결과로 발생한 '
 '신체장해나 재 물손해를 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 174},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000580',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
