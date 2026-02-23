from langchain_core.documents import Document

chunk = Document(
    page_content=("말합니다.</td></tr></tbody></table><footer id='9' "
 "style='font-size:14px'>47</footer><table id='10' "
 "style='font-size:16px'><thead><tr><td>용어</td><td>정의</td></tr></thead><tbody><tr><td>중요한 "
 '사항</td><td>계약 전 알릴 의무와 관련하여 회사가 그 사 실을 알았더라면 계약의 청약을 거절하거나 보험가입금액 한도 제한, 일부 '
 '보장 제외, 보 험금 삭감, 보험료 할증과 같이 조건부로 승 낙하는 등'),
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
 'indexing': {'chunk_id': 'chunk_000006',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
