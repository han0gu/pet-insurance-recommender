from langchain_core.documents import Document

chunk = Document(
    page_content=('명합니다.제9조(만기환급금의 지급)# \uf000 회사는 보험기간이끝난 때에 만기환급금을 보험수익자에게 지급합니다.- \uf000 회사는 '
 '계약자 및 보험수익자의 청구에 의하여 제1항에 의한 만기환급금을 지급하\n'
 '- 는 경우 청구일부터 3영업일 이내에 지급합니다.\n'
 '- 공\n'
 '- \uf000 회사는 제1항에 의한 만기환급금의 지급시기가 되면 지급시기 7일 이전에 그 사유\n'
 '- 통\n'
 '- 와 지급할 금액을 계약자 또는 보험수익자에게 알려드리며, 만기환급금을 지급함\n'
 '- 에 있어 지급일까지의 기간에 대한 이자의 계산은 "보험금을 지급할 때의 적립이율 사항'),
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
 'indexing': {'chunk_id': 'chunk_000035',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
