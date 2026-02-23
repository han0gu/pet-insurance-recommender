from langchain_core.documents import Document

chunk = Document(
    page_content=('. 도<br>성<br>1. 안면부란 이마를 포함하여 목까지의 얼굴부분을 말합니다.<br>특<br>2. 상지란 견관절 이하의 팔부분을 '
 '말합니다.<br>약<br>3'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000423',
              'chunk_char_len': 89,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
