from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 세균성 음식물 중 독과 상습적으로 흡입, 흡수 또는 섭취한 결과 로 생긴 중독증상은 이에 포함되지 '
 '않습니다.</td></tr><tr><td>질병</td><td>상해를 제외한 상병을 모두 포함합니다.</td></tr><tr><td>중요한 '
 '사항</td><td>계약 전 알릴 의무와 관련하여 회사가 그 사실 을 알았더라면 계약의 청약을 거절하거나 보험 가입금액 한도 제한, 일부 '
 '보장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000275',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
