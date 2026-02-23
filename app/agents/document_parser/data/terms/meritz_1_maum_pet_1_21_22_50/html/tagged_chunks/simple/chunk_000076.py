from langchain_core.documents import Document

chunk = Document(
    page_content=('남는 방법으로 회사가 알린 사항은 일반적으로 도달에 필요한 기간이 지<br>난 때에 계약자 또는 보험수익자에게 도달된 것으로 '
 "봅니다.</p><footer id='91' style='font-size:14px'>- 8 -</footer><h1 id='92' "
 "style='font-size:14px'>제13조(보험수익자의 지정)</h1><br><h1 id='93' "
 "style='font-size:14px'>보험수익자를 지정하지 않은 때에는 보험수익자를 피보험자로 합니다.</h1><h1 id='94'"),
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
 'indexing': {'chunk_id': 'chunk_000076',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
