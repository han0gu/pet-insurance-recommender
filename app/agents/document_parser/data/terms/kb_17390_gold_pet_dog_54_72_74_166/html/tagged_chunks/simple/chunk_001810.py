from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="25"></td><td rowspan="25">803 순환기계질환</td><td>질병 세부 '
 '질병명</td></tr><tr><td>심장사상충 '
 '감염</td></tr><tr><td>대동맥판폐쇄부전</td></tr><tr><td>대동맥협착증</td></tr><tr><td>동맥관개존증</td></tr><tr><td>전도'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001810',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
