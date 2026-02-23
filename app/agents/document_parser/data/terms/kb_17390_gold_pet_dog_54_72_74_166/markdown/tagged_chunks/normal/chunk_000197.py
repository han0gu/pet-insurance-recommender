from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 청구서(회사양식)\n'
 '- 2. 사고증명서\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증)\n'
 '- 4. 피보험자 및 지정대리청구인의 가족관계등록부 및 주민등록등본\n'
 '5. 기타 지정대리청구인이 보험금 등의 수령에 필요하여 제출하는 서류- 제42조(지정대리청구인에 의한 보험금의 지급 절차)\n'
 '- \uf000 지정대리청구인은 제41조(지정대리청구인에 의한 보험금의 청구)에 정한 구비서류\n'
 '- 를 제출하고 회사의 승낙을 얻어 제38조(적용대상)의 보험수익자의 대리인으로서'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000197',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
