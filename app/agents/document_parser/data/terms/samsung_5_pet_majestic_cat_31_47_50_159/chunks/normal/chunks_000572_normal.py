from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 회사는 제1항의 규정에 정한 지급기일 내에 보험금을 지급하지 않았을 때(제2항의 규 정에서 정한 지급예정일을 통지한 경우를 '
 '포함합니다)에는 그 다음날부터 지급일까지 의 기간에 대하여 보험금을 지급할 때의 적립이율 계산([별표1] 참조)에서 정한 이율 로 계산한 '
 '금액을 보험금에 더하여 지급합니다. 그러나 계약자, 피보험자 또는 보험수 익자의 책임있는 사유로 지급이 지연된 때에는 그 해당기간에 대한 '
 '이자는 더하여 지 급하지 않습니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000572',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
