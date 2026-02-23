from langchain_core.documents import Document

chunk = Document(
    page_content=('| 장해 | 【별표2(장해분류표)】에서 정한 기준에 따른 장해상태를 말합니다. |\n'
 '51| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 중요한 사항 | 계약 전 알릴 의무와 관련하여 회사가 그 사 실을 알았더라면 계약의 청약을 거절하거나 보험가입금액 한도 제한, 일부 '
 '보장 제외, 보 험금 삭감, 보험료 할증과 같이 조건부로 승 낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합니다. |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000003',
              'chunk_char_len': 220,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
