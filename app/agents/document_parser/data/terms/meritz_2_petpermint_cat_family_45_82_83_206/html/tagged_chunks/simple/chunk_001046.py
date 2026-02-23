from langchain_core.documents import Document

chunk = Document(
    page_content=(". 지급률의 결정</h1><br><p id='70' data-category='list' style='font-size:20px'>1) "
 '한 다리의 3대 관절중 관절 하나에 기능장해가 생기고<br>다른 관절 하나에 기능장해가 발생한 경우 지급률은<br>각각 적용하여 '
 '합산한다.<br>2) 1하지(다리와 발가락)의 장해 지급률은 원칙적으로 각<br>각 합산하되, 지급률은 60% 한도로 한다.</p><h1 '
 "id='71' style='font-size:20px'>10"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001046',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
