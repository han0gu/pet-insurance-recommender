from langchain_core.documents import Document

chunk = Document(
    page_content=('감소가 의심되지만 의사소<br>통이 되지 않는 경우, 만 3세 미만의 소아 포함) 검사결과에 대한 검증<br>이 필요한 경우에는 '
 "‘언어청력검사, 임피던스 청력검사, 청성뇌간반응</p><br><p id='185' data-category='paragraph' "
 "style='font-size:18px'>- 141 -</p><br><p id='186' data-category='paragraph' "
 "style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 141</p><br><p id='187'"),
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
 'indexing': {'chunk_id': 'chunk_001498',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
