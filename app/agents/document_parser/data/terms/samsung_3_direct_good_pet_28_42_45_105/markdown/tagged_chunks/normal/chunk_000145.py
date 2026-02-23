from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 서류를 접수한 날부터 30영업일 이내에서 정합니다.\n'
 '- 1. 소송제기\n'
 '- 2. 분쟁조정 신청\n'
 '- 3. 수사기관의 조사\n'
 '- 4. 해외에서 발생한 보험사고에 대한 조사\n'
 '- 5. 제6항에 따른 회사의 조사요청에 대한 동의 거부 등 계약자, 피보험자 또는 보험수\n'
 '- 익자의 책임있는 사유로 보험금 지급사유의 조사 및 확인이 지연되는 경우\n'
 '- 6. 각 특별약관별 보험금 지급에 관한 세부규정에 따라 보험금 지급사유에 대해 제3\n'
 '- 자의 의견에 따르기로 한 경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000145',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
