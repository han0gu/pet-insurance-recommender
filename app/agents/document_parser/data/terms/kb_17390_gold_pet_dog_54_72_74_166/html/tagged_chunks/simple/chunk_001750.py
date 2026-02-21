from langchain_core.documents import Document

chunk = Document(
    page_content=('표재성인 것</td><td></td></tr><tr><td>1) 길이 2.5cm '
 '미만</td><td>SC021</td></tr><tr><td>2) 길이 2.5cm 이상 ~ 5.0cm '
 "미만</td><td>SC022</td></tr></tbody></table><p id='75' "
 "data-category='paragraph' style='font-size:18px'>160 -</p><table id='76' "
 "style='font-size:16px'><thead><tr><td"),
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
 'indexing': {'chunk_id': 'chunk_001750',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
