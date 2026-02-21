from langchain_core.documents import Document

chunk = Document(
    page_content=("id='177' data-category='paragraph' style='font-size:14px'>및</p><p id='178' "
 "data-category='paragraph' style='font-size:16px'>제6조(보험료의 납입을 연체하여 해지된 계약의 "
 "부활(효력회복))</p><br><p id='179' data-category='paragraph' "
 "style='font-size:16px'>부활(효력회복)되는 계약의 보장개시는 반려동물(강아지) 일반조항</p><br><p "
 "id='180'"),
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
 'indexing': {'chunk_id': 'chunk_000983',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
