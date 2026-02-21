from langchain_core.documents import Document

chunk = Document(
    page_content=("id='10' style='font-size:18px'>지정대리청구서비스 특별약관</h1><h1 id='11' "
 "style='font-size:14px'>제1조(적용대상)</h1><br><p id='12' "
 "data-category='paragraph' style='font-size:14px'>이 특별약관(이하「특약」이라 합니다)은 "
 '보험계약자(이하「계약자」라 합니다), 피보험자<br>및 보험수익(이하「수익자」라 합니다)자가 모두 동일한 보통약관 및 특별약관에 '
 "적용됩니<br>다.</p><h1 id='13'"),
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
 'indexing': {'chunk_id': 'chunk_000343',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
