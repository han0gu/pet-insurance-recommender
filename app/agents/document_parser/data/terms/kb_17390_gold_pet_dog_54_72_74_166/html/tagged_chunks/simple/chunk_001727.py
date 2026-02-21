from langchain_core.documents import Document

chunk = Document(
    page_content=('. 상완골, 견갑골</td><td>N0972</td></tr><tr><td>다. 전완골, '
 '하퇴골</td><td>N0973</td></tr><tr><td>(1)요골과 척골중하나, 경골과 비골중 '
 '하나</td><td>N0977</td></tr><tr><td>(2)요척골 동시, 경비골 '
 '동시</td><td>N0974</td></tr><tr><td>라. 쇄골, 슬개골, 수근골, '
 '족근골</td><td>N0975</td></tr><tr><td>마'),
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
 'indexing': {'chunk_id': 'chunk_001727',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
