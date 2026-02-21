from langchain_core.documents import Document

chunk = Document(
    page_content=('② 새로이 증가된 보험의 목적의 보험기간이 시작된 후라도 다른 약정이 없으면 추가 보험\n'
 '료를 받기 전에 생긴 손해는 보상하여 드리지 않습니다.제4조(준용규정)이 추가특별약관에 정하지 않은 사항은 보통약관 및 단체계약 '
 '특별약관을 따릅니다.- 41 -지정대리청구서비스 특별약관제1조(적용대상)이 특별약관(이하「특약」이라 합니다)은 보험계약자(이하「계약자」라 '
 '합니다), 피보험자\n'
 '및 보험수익(이하「수익자」라 합니다)자가 모두 동일한 보통약관 및 특별약관에 적용됩니'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000188',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
