from langchain_core.documents import Document

chunk = Document(
    page_content=('. 전환대상계약의 피보험자 1인은 비장애인이고 보험수익자 2인 중 한명은 비장애인,<br>한명은 장애인인 경우</p><br><p '
 "id='54' data-category='paragraph' style='font-size:14px'>: 모든 보험수익자가 장애인이 "
 "아니므로 이 특별약관을 적용할 수 없습니다.</p><br><p id='55' data-category='paragraph' "
 "style='font-size:14px'>2"),
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
 'indexing': {'chunk_id': 'chunk_000373',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
