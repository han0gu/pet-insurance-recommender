from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>- 43 -</footer><h1 id='32' "
 "style='font-size:18px'>장애인전용보험전환 특별약관</h1><h1 id='33' "
 "style='font-size:14px'>제1조(특별약관의 적용범위)</h1><br><p id='34' "
 "data-category='paragraph' style='font-size:14px'>① 이 특별약관은 보험회사(이하「회사」라 "
 '합니다)가 정한 방법에 따라 보험계약자(이하<br>「계약자」라 합니다)가 청약하고 회사가 승낙함으로써 다음 각 호의'),
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
 'indexing': {'chunk_id': 'chunk_000358',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
