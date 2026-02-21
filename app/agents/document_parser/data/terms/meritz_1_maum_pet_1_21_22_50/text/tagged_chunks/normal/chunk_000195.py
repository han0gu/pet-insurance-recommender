from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 피보험자 및 지정대리청구인의 가족관계등록부(가족관계증명서) 및 주민등록등본\n'
 '5. 기타 지정대리청구인이 보험금 등의 수령에 필요하여 제출하는 서류제7조(준용규정)이 특약에서 정하지 않은 사항에 대하여는 보통약관 및 '
 '해당 특별약관을 따릅니다.- 43 -장애인전용보험전환 특별약관제1조(특별약관의 적용범위)① 이 특별약관은 보험회사(이하「회사」라 '
 '합니다)가 정한 방법에 따라 보험계약자(이하\n'
 '「계약자」라 합니다)가 청약하고 회사가 승낙함으로써 다음 각 호의 조건을 모두 만족'),
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
 'indexing': {'chunk_id': 'chunk_000195',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
