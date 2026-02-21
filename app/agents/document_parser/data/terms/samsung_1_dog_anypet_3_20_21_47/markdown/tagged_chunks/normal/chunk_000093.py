from langchain_core.documents import Document

chunk = Document(
    page_content=('그 지급을 보장합니다.【예금자보호제도】 예금보험공사에서 금융기관 등으로부터 미리 보험료를 받아 적립해 두었다가 금융기관이\n'
 '예금보험공사에서 금융기관 등으로부터 미리 보험료를 받아 적립해 두었다가 금융기관이 경영악화나 파산 등\n'
 '으로 예금을 지급할 수 없는 경우 해당 금융기관을 대신하여 예금자에게 보험금 또는 환급금을 지급함으로써\n'
 '예금자를 보호하는 제도를 말합니다. 이 보험계약은 예금자보호법에 따라 해약환급금(또는 만기 시 보험금)에'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000093',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
