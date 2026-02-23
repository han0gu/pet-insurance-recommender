from langchain_core.documents import Document

chunk = Document(
    page_content=('. 모든 피보험자 또는 모든 보험수익자가「소득세법 시행령 제107조(장애인의 범위) 제<br>1항」에서 규정한 장애인인 보험</p><p '
 "id='47' data-category='paragraph' style='font-size:14px'>【「소득세법 시행령 "
 "제107조(장애인의 범위) 제1항」에서 규정한 장애인】</p><br><p id='48' data-category='paragraph' "
 "style='font-size:14px'>① 법 제51조 제1항 제2호에 따른 장애인은 다음 각 호의 어느 하나에 해당하는"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000368',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
