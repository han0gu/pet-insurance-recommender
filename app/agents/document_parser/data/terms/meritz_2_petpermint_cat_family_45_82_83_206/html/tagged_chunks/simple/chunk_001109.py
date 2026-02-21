from langchain_core.documents import Document

chunk = Document(
    page_content=('보행이 불가능한 상태(20%) - 보조기구 없이 독립적인 보행은 가능하나 보행 시 파행(절뚝거림)이 있으며, 난간을 잡지 않 고는 계단을 '
 '오르내리기가 불가능한 상태 또는 평지에서 100m 이상을 걷지 못하는 상태(10%)</td></tr><tr><td>음식물 '
 '섭취</td><td>- 입으로 식사를 전혀 할수 없어 계속적으로 튜 브(비위관 또는 위루관)나 경정맥 수액을 통해 부분 혹은 전적인 '
 '영양공급을 받는 상태(20%) - 수저 사용이 불가능하여 다른 사람의 계속적 인 도움이 없이는 식사를 전혀 할 수 없는 상 태(15%) '
 '-'),
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
 'indexing': {'chunk_id': 'chunk_001109',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
