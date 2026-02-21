from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해판정기준</p><br><p id='183' data-category='paragraph' "
 "style='font-size:16px'>1)</p><br><p id='184' data-category='paragraph' "
 "style='font-size:16px'>신경계</p><br><h1 id='185' style='font-size:16px'>가) "
 "“신경계에 장해를 남긴 때”라 함은 뇌, 척수 및 말초신경계 손상으</h1><br><p id='186' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001652',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
