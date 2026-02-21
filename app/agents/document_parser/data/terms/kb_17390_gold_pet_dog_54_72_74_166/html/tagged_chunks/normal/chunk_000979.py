from langchain_core.documents import Document

chunk = Document(
    page_content=(". 검사 및 진단을 위한 수술(생검(生檢), 복강경검사(腹腔鏡檢査) 등)</p><br><p id='169' "
 "data-category='paragraph' style='font-size:14px'>별</p><br><p id='170' "
 "data-category='paragraph' style='font-size:14px'>약</p><p id='171' "
 "data-category='paragraph' style='font-size:14px'>해</p><table id='172'"),
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
 'indexing': {'chunk_id': 'chunk_000979',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
