from langchain_core.documents import Document

chunk = Document(
    page_content=('. 용 어 풀 이 해지</td></tr><tr><td>현재</td><td>있는 소멸시키</td></tr><tr><td '
 'colspan="2">유지되고 계약 또는 효력이 상실된 계약을 장래를 향하여 거나 계약유지 의사를 포기하여 만기일 이전에 계약관계를 '
 "청산하는 것</td></tr></tbody></table><br><p id='19' data-category='paragraph' "
 "style='font-size:16px'>제10조(사기에 의한 계약)</p><br><h1 id='20' "
 "style='font-size:16px'>\uf000 계약자"),
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
 'indexing': {'chunk_id': 'chunk_000852',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
