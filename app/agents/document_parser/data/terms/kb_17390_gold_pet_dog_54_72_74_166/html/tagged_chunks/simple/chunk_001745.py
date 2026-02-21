from langchain_core.documents import Document

chunk = Document(
    page_content=('~ 7.5cm 미만</td><td>SA038</td></tr><tr><td>5) 길이 7.5cm 이상 ~ 10.0cm '
 '미만</td><td>SA039</td></tr><tr><td>KB 금쪽같은 '
 "펫보험(강아지)(무배당)(26.01)</td><td>-</td></tr></tbody></table><br><table id='70' "
 "style='font-size:14px'><thead><tr><td></td></tr></thead><tbody><tr><td>대상이 "
 '되는 항목 수가코드 주: 길이 10cm이상 창상봉합술을 시행할경우'),
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
 'indexing': {'chunk_id': 'chunk_001745',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
