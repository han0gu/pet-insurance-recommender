from langchain_core.documents import Document

chunk = Document(
    page_content=('【제척기간】 어떤 종류의 권리에 대하여 법률상으로 정하여진 존속기간을 말하며, 이 기간이 지나면 해당 권리는 소멸됩니다.\n'
 '제30조(보험료의 환급)\n'
 '① 이 계약이 무효, 효력상실 또는 해지된 때에는 다음과 같이 보험료를 돌려드립니다.\n'
 '1. 계약자 또는 피보험자의 책임 없는 사유에 의하는 경우 : 무효의 경우에는 회사에 납입한 보험료 의 전액, 효력상실 또는 해지의 '
 '경우에는 경과하지 않은 기간에 대하여 일단위로 계산한 보험료'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000093',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
