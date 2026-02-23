from langchain_core.documents import Document

chunk = Document(
    page_content=('것</td><td></td></tr><tr><td>(가) 1) 길이 1.5cm 미만 '
 'SA021</td><td>특별</td></tr><tr><td>2) 길이 1.5cm 이상 ~ 3.0cm 미만 '
 "SA022</td><td>약</td></tr></tbody></table><br><p id='62' "
 "data-category='paragraph' style='font-size:14px'>관</p><p id='63' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001735',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
