from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약자(이하<br>「계약자」라 합니다)가 청약하고 회사가 승낙함으로써 다음 각 호의 조건을 모두 만족<br>하는 '
 '보험계약(이하「전환대상계약」이라 합니다)에 대하여 장애인전용보험으로 전환을<br>청약하는 경우에 적용합니다.</p><br><p '
 "id='35' data-category='paragraph' style='font-size:14px'>1"),
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
 'indexing': {'chunk_id': 'chunk_000359',
              'chunk_char_len': 194,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
