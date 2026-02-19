from langchain_core.documents import Document

chunk = Document(
    page_content=('. 3) “다리”라 함은 엉덩이관절(고관절)부터 발목관절 (족관절)까지를 말한다. 4) “다리의 3대 관절”이라 함은 '
 '엉덩이관절(고관절), 무 릎관절(슬관절), 발목관절(족관절)을 말한다. 5) “한다리의 발목이상을 잃었을 때”라 함은 발목관절 '
 '(족관절)부터(발목관절 포함) 심장에 가까운 쪽에서 절단된 때를 말하며, 무릎관절(슬관절)의 상부에서 절단된 경우도 포함한다. 6) '
 '다리의 관절기능 장해 평가는 다리의 3대관절의 관절'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 193},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'head']},
 'indexing': {'chunk_id': 'chunk_000700',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
