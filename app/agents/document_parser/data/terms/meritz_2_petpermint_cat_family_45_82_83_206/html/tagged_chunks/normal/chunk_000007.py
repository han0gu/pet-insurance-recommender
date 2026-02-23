from langchain_core.documents import Document

chunk = Document(
    page_content=('한도 제한, 일부 보장 제외, 보 험금 삭감, 보험료 할증과 같이 조건부로 승 낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 '
 '말합니다.</td></tr><tr><td>한국표준 질병사인 분류</td><td>제9차 개정 한국표준질병ㆍ사인분류(통계청 고시 '
 '제2025-299호, 2026.1.1.시행)를 말합니 다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000007',
              'chunk_char_len': 174,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
