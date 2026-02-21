from langchain_core.documents import Document

chunk = Document(
    page_content=('것이 어려운 상태 또는 배변, 배뇨는 독립적으로 가능하나 요실 금, 변실금이 있는 '
 '때(5%)</td></tr><tr><td>목욕</td><td>- 세안, 양치, 샤워, 목욕 등 모든 개인위생 관 리시 타인의 지속적인 '
 '도움이 필요한 상태 (10%) - 세안, 양치시 부분적인 도움 하에 혼자서 가능 하나 목욕이나 샤워시 타인의 도움이 필요한 상태(5%) '
 '- 세안, 양치와 같은 개인위생관리를 독립적으로 시행가능하나 목욕이나 샤워시 부분적으로 타 인의 도움이 필요한 '
 '상태(3%)</td></tr><tr><td>옷입고 벗기</td><td>-'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001113',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
