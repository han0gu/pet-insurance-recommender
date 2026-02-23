from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조(보험금의 청구)\n'
 '\uf000 피보험자 또는 지정대리청구인은 제1조에 정한 특별약관의 보험기간 중에 회사가\n'
 '정하는 바에 따라 다음의 서류를 제출하고 이 특별약관의 보험금을 청구하여야 합- 니다.\n'
 '- 1. 청구서(회사양식) 상\n'
 '- 2. 사고증명서(종합병원에서 발급한 진단서) 해\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증)\n'
 '- 4. 피보험자 및 지정대리청구인의 주민등록등본(지정대리청구인이 청구할 경우)\n'
 '5. 기타, 피보험자 또는 지정대리청구인이 보험금의 수령에 필요하여 제출하는 서류'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000765',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
