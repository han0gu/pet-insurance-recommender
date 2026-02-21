from langchain_core.documents import Document

chunk = Document(
    page_content=(". 주택 임차보증금의 반환을 보증하는 것을 목적으로 하는 보험·보증.</p><br><p id='80' "
 "data-category='paragraph' style='font-size:16px'>경우는 제외한다.</p><br><h1 "
 "id='81' style='font-size:16px'>다만, 보증대상 임차보증금이 3억원을 초과하는</h1><br><p id='82' "
 "data-category='paragraph' style='font-size:16px'>∙ 소득세법 시행규칙 제61조의3 "
 '(공제대상보험료의 범위)</p><br><p'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001414',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
