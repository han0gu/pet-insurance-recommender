from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 소송제기 2. 분쟁조정 신청 3. 수사기관의 조사 4. 해외에서 발생한 보험사고에 대한 조사 5. 제6항에 따른 회사의 조사요청에 '
 '대한 동의 거부 등 계약자, 피보험자 또는 보험수 익자의 책임있는 사유로 보험금 지급사유의 조사 및 확인이 지연되는 경우 6. '
 '제4조(보험금 지급에 관한 세부규정)에 따라 보험금 지급사유에 대해 제3자의 의견 에 따르기로 한 경우\n'
 '<유의사항>\n'
 '분쟁조정은 제36조(분쟁의 조정) 제1항에 따라 금융감독원에 신청할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 30},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000030',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
