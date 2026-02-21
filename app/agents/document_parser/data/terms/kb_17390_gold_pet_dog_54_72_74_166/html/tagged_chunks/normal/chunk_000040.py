from langchain_core.documents import Document

chunk = Document(
    page_content=(". 선박에 탑승하는 것을 직무로</h1><br><p id='49' data-category='paragraph' "
 "style='font-size:14px'>하는 사람이 직무상 선박에 탑승하고 있는 동안</p><br><p id='50' "
 "data-category='paragraph' style='font-size:14px'>제6조(보험금 지급사유의 통지)</p><br><p "
 "id='51' data-category='paragraph' style='font-size:14px'>계약자 또는 피보험자나 보험수익자는 "
 '제3조(보험금의 지급사유)에서'),
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
 'indexing': {'chunk_id': 'chunk_000040',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
