from langchain_core.documents import Document

chunk = Document(
    page_content=('피보험자가 사망하였을 경우에는 이 특별약관 계약도 소멸되며 회사는 "보험료 및\n'
 '해약환급금 산출방법서"에서 정하는 바에 따라 피보험자의 사망 당시 이 특별약관의\n'
 '계약자적립액 및 미경과보험료를 계약자에게 지급합니다.- \uf000 회사는 피보험자가 이 특별약관의 보험기간 중에 특정정신질환으로 진단 '
 '확정된\n'
 '- 경우에는 연간 1회에 한하여 이 특별약관의 보험가입금액을 특정정신질환진단비\n'
 '- 로 보험수익자에게 지급합니다.\n'
 '- \uf000 제1항에서 "연간"이란 계약일부터 매1년 단위로 도래하는 계약해당일 전일까지\n'
 '- 기간을 의미합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000424',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
