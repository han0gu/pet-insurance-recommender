from langchain_core.documents import Document

chunk = Document(
    page_content=('치아에 14개 이상의 결손이 생긴 때</td><td>20</td></tr><tr><td>9) 치아에 7개 이상의 결손이 생긴 '
 '때</td><td>10</td></tr><tr><td>10) 치아에 5개 이상의 결손이 생긴 '
 "때</td><td>5</td></tr></tbody></table><h1 id='211' style='font-size:14px'>나"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_001516',
              'chunk_char_len': 196,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
