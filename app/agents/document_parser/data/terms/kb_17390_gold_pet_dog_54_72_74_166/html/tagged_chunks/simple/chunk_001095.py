from langchain_core.documents import Document

chunk = Document(
    page_content=('"보험료 및 해약환급금 산출방법서"에서 정하는 바 약</p><br><p id=\'78\' '
 "data-category='paragraph' style='font-size:16px'>지급사유)에서 정한 무지개다리위로금(강아지, "
 "사망)을</p><br><p id='79' data-category='paragraph' "
 "style='font-size:14px'>제</p><p id='80' data-category='paragraph' "
 "style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01)"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001095',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
