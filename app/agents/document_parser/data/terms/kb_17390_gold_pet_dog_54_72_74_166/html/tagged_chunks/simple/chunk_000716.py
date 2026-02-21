from langchain_core.documents import Document

chunk = Document(
    page_content=("id='25' data-category='paragraph' style='font-size:14px'>제1조(보험금의 "
 "지급사유)</p><br><p id='26' data-category='paragraph' "
 "style='font-size:14px'>\uf000 회사는 피보험자가 이</p><br><p id='27' "
 "data-category='paragraph' style='font-size:14px'>특별약관의 보험기간 중에 진단확정된 질병으로 병원 "
 "또는</p><br><p id='28' data-category='list'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000716',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
