from langchain_core.documents import Document

chunk = Document(
    page_content=('제도로 피보험자가 필요로 하는 비용을 보전해 주기 위\n'
 '해 회사가 먼저 지급하는 임시 교부금을 말합니다.\uf000 회사는 제1항에서 정한 지급기일내에 보험금을 지급하지93않았을 때(제2항에서 '
 '정한 지급예정일을 통지한 경우를 포\n'
 '함합니다)에는 그 다음날부터 지급일까지의 기간에 대하여\n'
 '【별표1(보험금을 지급할 때의 적립이율 계산)】에서 정한\n'
 '이율로 계산한 금액을 보험금에 더하여 지급합니다. 그러나\n'
 '계약자, 피보험자 또는 보험수익자의 책임있는 사유로 지급\n'
 '이 지연된 때에는 그 해당기간에 대한 이자는 더하여 지급'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000164',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
