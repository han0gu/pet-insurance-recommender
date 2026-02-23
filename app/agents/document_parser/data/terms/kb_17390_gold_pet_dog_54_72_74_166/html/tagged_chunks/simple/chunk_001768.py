from langchain_core.documents import Document

chunk = Document(
    page_content=("금쪽같은 펫보험(강아지)(무배당)(26.01) 161</p><br><p id='89' data-category='paragraph' "
 "style='font-size:20px'>- 161 -</p><table id='90' "
 "style='font-size:14px'><thead><tr><td "
 'colspan="2"></td><td></td></tr></thead><tbody><tr><td '
 'rowspan="2">특정세균성</td><td>대상이 되는 항목</td><td>분류번호</td></tr><tr><td>폐렴연쇄알균에 의한 '
 '폐렴'),
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
 'indexing': {'chunk_id': 'chunk_001768',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
