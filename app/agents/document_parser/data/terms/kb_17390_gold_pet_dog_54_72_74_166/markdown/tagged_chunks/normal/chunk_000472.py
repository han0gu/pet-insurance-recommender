from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 분쟁조정 신청은 이 약관의 ｢분쟁의 조정｣ 조항에 따르며 분쟁조정 신청 대상기 관은 금융감독원의 금융분쟁조정위원회를 말합니다. | '
 '분쟁조정 신청은 이 약관의 ｢분쟁의 조정｣ 조항에 따르며 분쟁조정 신청 대상기 관은 금융감독원의 금융분쟁조정위원회를 말합니다. |\n'
 '\uf000 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에\n'
 '따라 회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 지급합니다.| 용 어 풀 이 | 가지급보험금 |\n'
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000472',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
