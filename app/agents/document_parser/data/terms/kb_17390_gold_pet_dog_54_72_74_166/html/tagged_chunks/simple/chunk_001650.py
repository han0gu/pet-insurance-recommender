from langchain_core.documents import Document

chunk = Document(
    page_content=('뚜렷한 치매 : CDR 척도 3점</td><td>60 특별</td></tr><tr><td>10) 약간의 치매 : CDR 척도 '
 '2점</td><td>약 40 관</td></tr><tr><td>11) 심한 뇌전증 발작이 남았을 '
 '때</td><td>70</td></tr><tr><td>12) 뚜렷한 뇌전증 발작이 남았을 '
 '때</td><td>40</td></tr><tr><td>13) 약간의 뇌전증 발작이 남았을 '
 "때</td><td>10</td></tr></tbody></table><p id='182' data-category='list'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_001650',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
