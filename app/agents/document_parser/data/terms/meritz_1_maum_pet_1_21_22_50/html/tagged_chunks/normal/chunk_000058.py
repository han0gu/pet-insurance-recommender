from langchain_core.documents import Document

chunk = Document(
    page_content=("수족관의 동물<br>에게 투여할 목적으로 처방대상 동물용 의약품에 대한 처방전을 발급할 수 있</p><footer id='63' "
 "style='font-size:14px'>- 6 -</footer><p id='64' data-category='paragraph' "
 "style='font-size:14px'>다"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000058',
              'chunk_char_len': 170,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
