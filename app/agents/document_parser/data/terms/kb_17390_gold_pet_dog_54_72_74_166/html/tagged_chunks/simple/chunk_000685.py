from langchain_core.documents import Document

chunk = Document(
    page_content=(". 환경성질환입원일당(1일이상)</p><h1 id='247' style='font-size:16px'>제1조(보험금의 "
 "지급사유)</h1><br><h1 id='248' style='font-size:16px'>\uf000 회사는 "
 "피보험자가</h1><br><p id='249' data-category='paragraph' style='font-size:16px'>이 "
 '특별약관의 보험기간 중에 "환경성질환"으로 진단 확정되</p><br><p id=\'250\' '
 "data-category='paragraph' style='font-size:16px'>고"),
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
 'indexing': {'chunk_id': 'chunk_000685',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
