from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5. 제5항에 따른 회사의 조사요청에 대한 동의 거부 등 계약자, 피보험자 또는 보\n'
 '- 험수익자의 책임있는 사유로 보험금 지급사유의 조사와 확인이 지연되는 경우\n'
 '6. 보험금 지급사유에 대해 제3자의 의견에 따르기로 한 경우| 용 어 풀 | 이 분쟁조정 신청 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000471',
              'chunk_char_len': 161,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
