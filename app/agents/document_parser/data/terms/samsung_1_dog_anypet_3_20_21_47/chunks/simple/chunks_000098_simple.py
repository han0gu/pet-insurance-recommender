from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자 또는 피보험자의 책임 없는 사유에 의하는 경우 : 무효의 경우에는 회사에 납입한 보험료 의 전액, 효력상실 또는 해지의 '
 '경우에는 경과하지 않은 기간에 대하여 일단위로 계산한 보험료 2. 계약자 또는 피보험자의 책임 있는 사유에 의하는 경우 : 이미 경과한 '
 '기간에 대하여 단기요율 (1년미만의 기간에 적용되는 요율)로 계산한 보험료를 뺀 잔액. 다만, 계약자, 피보험자의 고의 또는 중대한 '
 '과실로 무효가 된 때에는 보험료를 돌려드리지 않습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
