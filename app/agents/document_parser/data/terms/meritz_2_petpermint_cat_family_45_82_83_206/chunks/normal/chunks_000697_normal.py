from langchain_core.documents import Document

chunk = Document(
    page_content=('2) 1상지(팔과 손가락)의 장해 지급률은 원칙적으로 각각 합산하되, 지급률은 60% 한도로 한다.\n'
 '9. 다리의 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 두다리의 발목이상을 잃었을 때 | 100\n'
 '2) 한다리의 발목이상을 잃었을 때 | 60\n'
 '3) 한다리의 3대관절중 관절 하나의 기능을 완전히 잃었 을 때 | 30\n'
 '4) 한다리의 3대관절중 관절 하나의 기능에 심한 장해 를 남긴 때 | 20\n'
 '5) 한다리의 3대관절중 관절 하나의 기능에 뚜렷한 장해 를 남긴 때 | 10'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 193},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000697',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
